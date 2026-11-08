"""Restart-safe continuation of the original HAR CNN-LSTM experiment.

Source checkpoint and held-out split remain immutable. The original notebook
console figures are not represented as CSVLogger training history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .app_support import load_classes
from .cache import sample_cached
from .train import load_splits


def checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=path.parent,
                                      prefix=".har_", suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
        json.dump(obj, f, indent=2)
        f.write("\n")
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def check_resume_source(source: Path, labels_path: Path, labels: list[str],
                        output: Path, initial_epoch: int | None) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Existing training checkpoint missing: {source}")
    if not labels_path.is_file() or load_classes(labels_path) != labels:
        raise ValueError("Training class ordering does not match checkpoint metadata")
    if source.resolve() == (output / "best.keras").resolve():
        raise ValueError("Output cannot overwrite the source checkpoint")
    if initial_epoch is None or initial_epoch < 1:
        raise ValueError("Explicit positive --initial-epoch required for resumed training")


def run(args) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_cache_mb <= 0:
        raise ValueError("Positive epochs, batch size and cache budget required")
    df, labels = load_splits(args.splits, args.data_root)
    root, output = Path(args.data_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.resume_from).resolve()
    mapping = Path(args.class_map)
    state_path = output / "training_state.json"
    old_state = json.loads(state_path.read_text()) if state_path.exists() else None
    original_epoch = args.initial_epoch
    if old_state is not None:
        original_epoch = old_state["initial_epoch"]
    check_resume_source(source, mapping, labels, output, original_epoch)
    signature = {
        "checkpoint_source_sha256": checksum(source),
        "splits_sha256": checksum(args.splits),
        "class_order": labels,
        "initial_epoch": original_epoch,
        "target_epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
    }
    if old_state:
        if any(old_state.get(k) != v for k, v in signature.items()):
            raise ValueError("Existing run has a different source, split, class mapping or settings")
        if old_state.get("fit_completed") and not args.evaluate_only:
            print("Training already completed in this output directory; no retraining.")
            return
        if (not args.evaluate_only and old_state.get("last_completed_epoch", original_epoch) > original_epoch
                and not (output / "backup").exists()):
            raise ValueError("Training backup missing; cannot pretend later epochs were restored")
    elif any((output / name).exists() for name in ("best.keras", "history.csv", "backup")):
        raise ValueError("Output contains untracked artifacts; choose a new output directory")
    if args.epochs <= original_epoch:
        raise ValueError("Target --epochs must exceed the original completed epoch")

    import tensorflow as tf
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    class VideoSequence(tf.keras.utils.Sequence):
        def __init__(self, subset: pd.DataFrame, shuffle: bool):
            super().__init__()
            self.items = subset.reset_index(drop=True)
            self.shuffle = shuffle
            self.indices = np.arange(len(self.items))
            self.on_epoch_end()

        def __len__(self):
            return (len(self.indices) + args.batch_size - 1) // args.batch_size

        def __getitem__(self, n: int):
            batch = self.items.iloc[self.indices[n * args.batch_size:(n + 1) * args.batch_size]]
            x = np.stack([sample_cached(root / p, cache_dir=args.cache_dir,
                                         limit_mb=args.max_cache_mb)
                          for p in batch.video_path], axis=0)
            return x, batch.target.to_numpy(dtype=np.int64)

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)

    best_path = output / "best.keras"
    if args.evaluate_only:
        if not old_state or not old_state.get("fit_completed") or not best_path.is_file():
            raise ValueError("Finish fitting and save best.keras before test evaluation")
        if (output / "test_metrics.json").exists():
            print("Held-out metrics already exist; test evaluation will not repeat.")
            return
        from sklearn.metrics import classification_report, confusion_matrix
        model = tf.keras.models.load_model(best_path, compile=False, safe_mode=True)
        test_frame = df[df.split == "test"].reset_index(drop=True)
        probabilities = model.predict(VideoSequence(test_frame, False), verbose=1)
        predicted = probabilities.argmax(axis=1)
        targets = test_frame.target.to_numpy(dtype=np.int64)
        report = classification_report(targets, predicted, labels=list(range(len(labels))),
                                       target_names=labels, zero_division=0, output_dict=True)
        report["test_accuracy"] = float(np.mean(predicted == targets))
        report["test_size"] = int(len(targets))
        write_json_atomic(output / "test_metrics.json", report)
        pd.DataFrame(confusion_matrix(targets, predicted, labels=list(range(len(labels)))),
                     index=labels, columns=labels).to_csv(output / "confusion_matrix.csv")
        old_state["test_evaluated_once"] = True
        write_json_atomic(state_path, old_state)
        print("Final held-out accuracy:", report["test_accuracy"])
        return

    # Restore Adam state when possible. Fallback is clearly labeled fine-tuning.
    try:
        model = tf.keras.models.load_model(source, compile=True, safe_mode=True)
        optimizer = getattr(model, "optimizer", None)
        if optimizer is None:
            raise ValueError("No optimizer in saved checkpoint")
        mode = "optimizer_state_loaded"
    except (ValueError, TypeError) as exc:
        model = tf.keras.models.load_model(source, compile=False, safe_mode=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        mode = "weights_only_finetuning"
        print("Optimizer not restorable; fine-tuning saved weights:", str(exc)[:250])
    if tuple(model.input_shape[1:]) != (20, 64, 64, 3) or model.output_shape[-1] != len(labels):
        raise ValueError("Source tensor shape does not match data/labels")
    if model.name != "cnn_lstm_3":
        raise ValueError("Expected the original cnn_lstm_3 checkpoint")

    info = dict(signature, resume_mode=mode, original_checkpoint_preserved=True,
                last_completed_epoch=(old_state or {}).get("last_completed_epoch", original_epoch),
                fit_completed=False, test_evaluated_once=False,
                status="training", optimizer_iterations_at_load=int(model.optimizer.iterations.numpy()))
    write_json_atomic(state_path, info)
    (output / "classes.json").write_text(json.dumps(labels, indent=2) + "\n")

    class RecordProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            info["last_completed_epoch"] = epoch + 1
            info["last_epoch_metrics"] = {k: float(v) for k, v in (logs or {}).items()}
            write_json_atomic(state_path, info)

    callbacks = [
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(output / "backup"), save_freq="epoch"),
        tf.keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(output / "history.csv"), append=bool(old_state)),
        tf.keras.callbacks.TerminateOnNaN(),
        RecordProgress(),
    ]
    train = VideoSequence(df[df.split == "train"], True)
    val = VideoSequence(df[df.split == "val"], False)
    try:
        model.fit(train, validation_data=val, initial_epoch=original_epoch,
                  epochs=args.epochs, callbacks=callbacks, verbose=1)
        if not best_path.is_file():
            raise RuntimeError("No validation checkpoint was saved; fitting is incomplete")
        model.save(output / "final.keras")
        info["status"] = "fit_completed"
        info["fit_completed"] = True
        write_json_atomic(state_path, info)
        print("Fit completed; use --evaluate-only for a single final held-out evaluation")
    except BaseException:
        info["status"] = "interrupted"
        write_json_atomic(state_path, info)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", default="data/train_val_test_splits_70_15_15.csv")
    parser.add_argument("--resume-from", required=True, help="Original checkpoint; never overwritten")
    parser.add_argument("--class-map", default="artifacts/classes.json")
    parser.add_argument("--initial-epoch", type=int, help="Previously completed source epochs, e.g. 3")
    parser.add_argument("--epochs", type=int, default=25, help="Total target epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, help="A NEW persistent output directory")
    parser.add_argument("--cache-dir", default="/content/har_preprocessed")
    parser.add_argument("--max-cache-mb", type=int, default=256)
    parser.add_argument("--evaluate-only", action="store_true", help="Final held-out evaluation, once")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

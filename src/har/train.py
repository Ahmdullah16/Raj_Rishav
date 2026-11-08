"""Memory-bounded training and single-use held-out test evaluation."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .model import build_model
from .video import sample_video


def load_splits(csv_path: str | Path, root: str | Path):
    frame = pd.read_csv(csv_path)
    required = {"video_path", "activity", "bytes", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Split CSV missing fields {required - set(frame.columns)}")
    if frame.video_path.duplicated().any() or not set(frame.split).issubset({"train", "val", "test"}):
        raise ValueError("Duplicates or invalid split names")
    labels = sorted(frame.activity.unique().tolist())
    if not all((frame.split == split).any() for split in ("train", "val", "test")):
        raise ValueError("Every split must be nonempty")
    root = Path(root).resolve()
    for rel in frame.video_path:
        candidate = (root / rel).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise FileNotFoundError(f"Missing or invalid video file: {candidate}")
    frame["target"] = frame.activity.map({label: i for i, label in enumerate(labels)})
    return frame, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Directory containing the dataset root folder")
    parser.add_argument("--splits", default="data/train_val_test_splits_70_15_15.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--variant", default="cnn_lstm_3", choices=("cnn_lstm_1", "cnn_lstm_2", "cnn_lstm_3", "mobilenetv2_lstm"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-test", action="store_true", help="Only for final evaluation after model selection")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("Epochs and batch size must be positive")
    import tensorflow as tf
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    df, labels = load_splits(args.splits, args.data_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)

    class VideoSequence(tf.keras.utils.Sequence):
        def __init__(self, subset, shuffle):
            super().__init__()
            self.items = subset.reset_index(drop=True)
            self.shuffle = shuffle
            self.indices = np.arange(len(self.items))
            self.on_epoch_end()
        def __len__(self):
            return int(np.ceil(len(self.indices) / args.batch_size))
        def __getitem__(self, i):
            selected = self.indices[i*args.batch_size:(i+1)*args.batch_size]
            batch = self.items.iloc[selected]
            X = np.stack([sample_video(root / p) for p in batch.video_path], axis=0)
            return X, batch.target.to_numpy(dtype=np.int64)
        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)

    train = VideoSequence(df[df.split == "train"], True)
    val = VideoSequence(df[df.split == "val"], False)
    model = build_model(len(labels), args.variant)
    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(output / "best.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(output / "history.csv")),
    ]
    history = model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks)
    model.save(output / "final.keras")
    (output / "classes.json").write_text(json.dumps(labels, indent=2))
    metadata = {"variant": args.variant, "classes": labels, "seed": args.seed,
                "dataset_rows": len(df), "split_counts": df.split.value_counts().to_dict(),
                "best_validation_accuracy": float(max(history.history["val_accuracy"]))}
    if args.evaluate_test:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        test = VideoSequence(df[df.split == "test"], False)
        test_model = tf.keras.models.load_model(output / "best.keras")
        prob = test_model.predict(test)
        pred = prob.argmax(axis=1)
        truth = df[df.split == "test"].target.to_numpy()
        metrics = classification_report(truth, pred, labels=list(range(len(labels))),
                                        target_names=labels, output_dict=True, zero_division=0)
        metrics["test_accuracy"] = float(accuracy_score(truth, pred))
        (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        pd.DataFrame(confusion_matrix(truth, pred, labels=list(range(len(labels)))),
                     index=labels, columns=labels).to_csv(output / "confusion_matrix.csv")
        metadata["test_evaluated_once"] = True
    else:
        metadata["test_evaluated_once"] = False
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Saved trained checkpoints and metrics to", output)


if __name__ == "__main__":
    main()

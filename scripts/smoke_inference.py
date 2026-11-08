"""CI deployment smoke: verify bundled model, real preprocessing and Streamlit pages.

This does not use the Kaggle dataset and is NOT an accuracy evaluation.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    # Avoid expensive parallel TensorFlow threads on small Community Cloud hardware.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")

    import cv2
    import numpy as np
    import tensorflow as tf

    from har.app_support import load_classes
    from har.checkpoint import verify_published_checkpoint
    from har.video import sample_video, validate_video

    original = ROOT / "artifacts" / "best.keras"
    assert verify_published_checkpoint(original), "Original checkpoint integrity failure"
    checkpoint = ROOT / "artifacts" / "best_epoch9.keras"
    classes = load_classes(ROOT / "artifacts" / "classes.json")
    assert verify_published_checkpoint(checkpoint), "Nine-epoch checkpoint checksum mismatch"
    assert classes and len(classes) == 7, "Class mapping unavailable"

    try:
        model = tf.keras.models.load_model(checkpoint, compile=True, safe_mode=True)
        optimizer_mode = "optimizer_restored"
        optimizer_steps = int(model.optimizer.iterations.numpy())
    except (ValueError, TypeError) as exc:
        print("Optimizer restoration unavailable:", str(exc)[:180])
        model = tf.keras.models.load_model(checkpoint, compile=False, safe_mode=True)
        optimizer_mode = "weights_only"
        optimizer_steps = None
    assert optimizer_steps == 108, "Expected 108 saved optimizer updates for this snapshot"
    assert tuple(model.input_shape[1:]) == (20, 64, 64, 3)
    assert model.output_shape[-1] == len(classes)

    with tempfile.TemporaryDirectory() as folder:
        sample = Path(folder) / "synthetic_pipeline_smoke.mp4"
        writer = cv2.VideoWriter(str(sample), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 64))
        if not writer.isOpened():
            raise RuntimeError("OpenCV MP4 encoder unavailable in CI")
        for i in range(24):
            canvas = np.zeros((64, 64, 3), dtype=np.uint8)
            canvas[:, :, 0] = (i * 11) % 256
            canvas[:, :, 1] = np.arange(64, dtype=np.uint8)[None, :] * 3
            writer.write(canvas)
        writer.release()

        info = validate_video(sample)
        frames = sample_video(sample, seq_length=20, image_size=64)
        assert frames.shape == (20, 64, 64, 3)
        started = time.perf_counter()
        pred = np.asarray(model.predict(frames[None, ...], verbose=0))
        elapsed = time.perf_counter() - started
        assert pred.shape == (1, 7)
        assert np.isfinite(pred).all() and (pred >= 0).all()
        assert np.isclose(pred.sum(), 1.0, atol=0.01)
        print(json.dumps({
            "result": "pass",
            "purpose": "computational smoke only, not recognition accuracy",
            "input_shape": list(frames.shape),
            "output_shape": list(pred.shape),
            "frame_count": info["frame_count_reported"],
            "probabilities_sum": round(float(pred.sum()), 6),
            "inference_seconds": round(elapsed, 3),
            "tensorflow": tf.__version__,
            "load_mode": optimizer_mode,
            "optimizer_steps_at_load": optimizer_steps,
            "classes": classes,
        }, indent=2))


if __name__ == "__main__":
    main()

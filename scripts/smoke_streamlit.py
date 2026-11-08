"""CI Streamlit UI smoke with real, tiny uploaded videos and pinned CNN-LSTM inference.

Exercises the actual file-uploader widget and predict button via Streamlit AppTest.
Uses GENERATED video bytes only; it is not a real-user browser test or an
accuracy measurement and does not access/download the research dataset.
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def assert_no_exception(app: AppTest, stage: str) -> None:
    if app.exception:
        raise AssertionError(f"{stage}: {[e.message for e in app.exception]}")


def make_short_mp4(folder: Path) -> bytes:
    path = folder / "generated_smoke.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 64)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV MP4 encoder unavailable in CI")
    try:
        for i in range(24):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 11) % 256
            frame[:, :, 1] = np.arange(64, dtype=np.uint8)[None, :] * 3
            writer.write(frame)
    finally:
        writer.release()
    content = path.read_bytes()
    if not content:
        raise AssertionError("Synthetic MP4 was not written")
    return content


def assert_prediction(app: AppTest, description: str) -> None:
    action = next(
        (button for button in app.button
         if button.label == "Analyze video and predict activity"), None
    )
    if action is None:
        raise AssertionError(f"{description}: prediction button unavailable")
    action.click()
    app.run(timeout=120)
    assert_no_exception(app, description)
    if app.error:
        raise AssertionError(f"{description}: {[e.value for e in app.error]}")
    if not any("Predicted activity:" in item.value
               for item in app.success):
        raise AssertionError(f"{description}: no model-generated prediction rendered")
    print(f"{description}: uploaded video decoded and predicted successfully.")


def main() -> None:
    app = AppTest.from_file(str(ROOT / "streamlit.py"), default_timeout=120)
    app.run()
    assert_no_exception(app, "Prediction landing page")
    if not app.sidebar.radio:
        raise AssertionError("Sidebar navigation did not render")

    app.sidebar.radio[0].set_value("Overview")
    app.run()
    assert_no_exception(app, "Overview")

    app.sidebar.radio[0].set_value("Predict Activity")
    app.run()
    assert_no_exception(app, "Predict Activity")
    if not app.file_uploader:
        raise AssertionError("Prediction page did not render a file uploader")

    with tempfile.TemporaryDirectory(prefix="har_app_smoke_") as folder:
        video = make_short_mp4(Path(folder))
        uploader = app.file_uploader(key="inference_video")
        uploader.upload("generated_smoke.mp4", video, "video/mp4")
        app.run()
        assert_no_exception(app, "Raw MP4 upload")
        assert_prediction(app, "Raw MP4")

        wrapped = io.BytesIO()
        with zipfile.ZipFile(wrapped, mode="w") as archive:
            archive.writestr("inside/video.mp4", video)
        uploader = app.file_uploader(key="inference_video")
        uploader.set_value(("wrapped.mp4", wrapped.getvalue(), "video/mp4"))
        app.run()
        assert_no_exception(app, "ZIP-wrapped MP4 upload")
        assert_prediction(app, "ZIP-wrapped MP4")
        if not any("ZIP-wrapped video" in item.value for item in app.caption):
            raise AssertionError("ZIP-wrapped video was not identified in UI")

    print("Streamlit AppTest video-upload and prediction smoke passed.")
    print("This is local widget testing, not anonymous public-browser verification.")


if __name__ == "__main__":
    main()

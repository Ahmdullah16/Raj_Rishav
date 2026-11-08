"""Anonymous *browser* diagnostic of the deployed HAR Streamlit prediction page.

Read-only with respect to external accounts: opens the app in a fresh, signed-out
Chromium browser and uploads a small generated MP4 for functional testing.
Does not touch Kaggle data, credentials, the existing model, or test metrics.

Results are intentionally nonblocking. A sleeping app, login interstitial, or
network error is NOT an acceptable scientific accuracy measurement.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

APP_URL = "https://rajrishav-har.streamlit.app/"
RESULT_PATH = Path(os.environ.get("HAR_BROWSER_REPORT", "public_browser_diagnostic.json"))
SCREENSHOT_PATH = Path(os.environ.get("HAR_BROWSER_SCREENSHOT", "public_browser_snapshot.png"))


def classify_observation(url: str, *, prediction_visible: bool = False,
                         uploader_visible: bool = False) -> str:
    """Classify what was actually observed, not what a plain HTTP redirect implies."""
    if prediction_visible:
        return "prediction_ui_verified"
    if uploader_visible:
        return "prediction_page_visible_upload_not_verified"
    host = urlsplit(url).netloc.lower()
    if host == "share.streamlit.io" or "/-/auth/" in url or "/-/login" in url:
        return "browser_auth_interstitial"
    return "prediction_page_not_observed"


def generate_mp4(path: Path) -> bytes:
    """Small deterministic clip; never treat its prediction as accuracy."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 64))
    if not writer.isOpened():
        raise RuntimeError("MP4 encoder unavailable on browser CI runner")
    try:
        for index in range(24):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = (index * 11) % 256
            frame[:, :, 1] = np.arange(64, dtype=np.uint8)[None, :] * 3
            writer.write(frame)
    finally:
        writer.release()
    return path.read_bytes()


def browser_probe() -> dict:
    from playwright.sync_api import sync_playwright

    result = {
        "app_url": APP_URL,
        "method": "signed-out Chromium browser; generated test video",
        "verified": False,
        "actual_dataset_used": False,
        "classification": "browser_not_started",
    }
    with TemporaryDirectory(prefix="har_public_browser_") as temporary:
        synthetic_mp4 = generate_mp4(Path(temporary) / "generated.mp4")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(viewport={"width": 1280, "height": 900})
            page = context.new_page()
            try:
                page.goto(APP_URL, wait_until="domcontentloaded", timeout=75000)
                # Allow cold Community Cloud app startup; do not authenticate.
                deadline = time.monotonic() + 140
                upload = page.locator('input[type="file"]').last
                while time.monotonic() < deadline:
                    if upload.count() > 0:
                        break
                    page.wait_for_timeout(1500)

                result["final_browser_host"] = urlsplit(page.url).netloc
                result["prediction_page_rendered"] = upload.count() > 0
                if upload.count() == 0:
                    result["classification"] = classify_observation(page.url)
                    return result

                result["classification"] = classify_observation(page.url, uploader_visible=True)
                upload.set_input_files({
                    "name": "generated_public_smoke.mp4",
                    "mimeType": "video/mp4",
                    "buffer": synthetic_mp4,
                })
                action = page.get_by_role("button", name="Analyze video and predict activity")
                action.wait_for(state="visible", timeout=60000)
                # A model download must not be needed: the verified archive is bundled.
                action.click(timeout=60000)
                alert = page.locator('[data-testid="stAlert"]').filter(has_text="Predicted activity:")
                alert.wait_for(state="visible", timeout=180000)
                has_csv = page.get_by_text("Download prediction CSV").count() > 0
                result.update(
                    classification=classify_observation(page.url, prediction_visible=True),
                    prediction_visible=True,
                    csv_download_control_visible=has_csv,
                    verified=bool(has_csv),
                )
                return result
            except Exception as exc:
                result["error_type"] = type(exc).__name__
                result["error_excerpt"] = str(exc).splitlines()[0][:220]
                result["final_browser_host"] = urlsplit(page.url).netloc
                result["classification"] = classify_observation(
                    page.url, uploader_visible=bool(result.get("prediction_page_rendered"))
                )
                return result
            finally:
                try:
                    page.screenshot(path=str(SCREENSHOT_PATH), full_page=False, timeout=8000)
                except Exception:
                    pass
                context.close()
                browser.close()


def main() -> int:
    try:
        result = browser_probe()
    except Exception as exc:
        result = {
            "app_url": APP_URL,
            "verified": False,
            "classification": "browser_probe_setup_failed",
            "error_type": type(exc).__name__,
            "error_excerpt": str(exc).splitlines()[0][:220],
        }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("This is a synthetic upload smoke, not accuracy or a real-human-video test.")
    return 0  # diagnostic: never make unrelated source checks fail for cloud access


if __name__ == "__main__":
    raise SystemExit(main())

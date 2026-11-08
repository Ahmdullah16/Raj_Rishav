"""Browser-observation classifications cannot equate HTTP redirects with privacy."""
from __future__ import annotations

import runpy
from pathlib import Path

_probe = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "probe_public_browser.py")
)
classify_observation = _probe["classify_observation"]


def test_actual_prediction_is_verified():
    assert classify_observation(
        "https://rajrishav-har.streamlit.app/", prediction_visible=True
    ) == "prediction_ui_verified"


def test_upload_page_does_not_imply_success():
    assert classify_observation(
        "https://rajrishav-har.streamlit.app/", uploader_visible=True
    ) == "prediction_page_visible_upload_not_verified"


def test_browser_auth_interstitial_is_reported_not_guessed():
    assert classify_observation(
        "https://share.streamlit.io/-/auth/app"
    ) == "browser_auth_interstitial"


def test_unseen_app_is_inconclusive():
    assert classify_observation(
        "https://rajrishav-har.streamlit.app/"
    ) == "prediction_page_not_observed"

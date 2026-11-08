"""Offline checks: a Streamlit session redirect is not proof of private access."""
from __future__ import annotations

import io
import runpy
import urllib.error
from pathlib import Path

_source = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "probe_public.py"))
URL, probe = _source["URL"], _source["probe"]


class StubOpener:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def open(self, request, timeout=15):
        assert request.full_url == URL
        if self.error:
            raise self.error
        return self.response


class StubResponse:
    status = 200

    def read(self, size):
        return b"ok"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_healthy_http_does_not_claim_browser_upload():
    actual = probe(StubOpener(response=StubResponse()))
    assert actual["verified"] is True
    assert actual["classification"] == "http_health_ok_browser_unchecked"
    assert actual["browser_upload_verified"] is False


def test_streamlit_auth_redirect_is_inconclusive_not_private():
    err = urllib.error.HTTPError(
        URL, 303, "See Other",
        {"Location": "https://share.streamlit.io/-/auth/app"},
        io.BytesIO(b""),
    )
    actual = probe(StubOpener(error=err))
    assert actual["status"] == 303
    assert actual["classification"] == "redirect_browser_check_required"
    assert actual["redirect_host"] == "share.streamlit.io"
    assert actual["verified"] is False
    assert actual["browser_upload_verified"] is False

"""Read-only HTTP health diagnostic for Community Cloud (not a browser test).

Some *public* Streamlit apps send non-browser clients through an auth/session
redirect. An HTTP 303 alone therefore does NOT prove app privacy or failure.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit

APP = "https://rajrishav-har.streamlit.app"
URL = APP + "/_stcore/health"
REDIRECTS = {301, 302, 303, 307, 308}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def probe(opener=None) -> dict:
    """Record one non-browser HTTP result; never infer browser accessibility."""
    result = {
        "url": APP,
        "check": "non-browser HTTP health only",
        "verified": False,
        "browser_upload_verified": False,
    }
    opener = opener or urllib.request.build_opener(NoRedirect())
    request = urllib.request.Request(URL, headers={"User-Agent": "HAR-HTTP-Health/1.0"})
    try:
        with opener.open(request, timeout=15) as response:
            body = response.read(80).decode("utf-8", errors="replace").strip()
            result.update(
                status=response.status,
                response=body,
                verified=response.status == 200 and body.lower() == "ok",
                classification=(
                    "http_health_ok_browser_unchecked"
                    if response.status == 200 and body.lower() == "ok"
                    else "unexpected_http_response"
                ),
            )
    except urllib.error.HTTPError as exc:
        result["status"] = exc.code
        result["reason"] = str(exc.reason)[:100]
        if exc.code in REDIRECTS:
            location = exc.headers.get("Location", "")
            result.update(
                classification="redirect_browser_check_required",
                redirect_host=urlsplit(location).netloc if location else None,
            )
        else:
            result["classification"] = "http_error_browser_check_required"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result.update(
            status="unavailable",
            classification="network_unavailable_browser_check_required",
            error_type=type(exc).__name__,
        )
    return result


def main() -> int:
    print(json.dumps(probe(), indent=2))
    print("HTTP redirects do not prove app privacy. Check via an anonymous browser.")
    print("The probe cannot verify upload or model-generated predictions.")
    return 0  # diagnostic only; sleeping or private apps do not fail source CI


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only public Streamlit health probe. Not a real browser upload test."""
import json
import urllib.error
import urllib.request

APP = "https://rajrishav-har.streamlit.app"
URL = APP + "/_stcore/health"


def main():
    result = {"url": APP, "check": "Streamlit HTTP health", "verified": False}
    try:
        with urllib.request.urlopen(URL, timeout=15) as response:
            body = response.read(80).decode("utf8", errors="replace").strip()
            result.update(status=response.status, response=body,
                          verified=response.status == 200 and body.lower() == "ok")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result["error_type"] = type(exc).__name__
        result["status"] = "unavailable"
    print(json.dumps(result, indent=2))
    print("Note: health does not prove that a public uploaded video was inferred.")
    # An app may be hibernated; record without failing source CI.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

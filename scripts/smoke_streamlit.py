"""Run Streamlit app's actual Overview and Predict Activity pages via AppTest."""
from __future__ import annotations

from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(ROOT / "streamlit.py"), default_timeout=90)
app.run()
if app.exception:
    raise AssertionError(f"Overview failed: {[e.message for e in app.exception]}")
if not app.sidebar.radio:
    raise AssertionError("Sidebar navigation did not render")
app.sidebar.radio[0].set_value("Predict Activity").run()
if app.exception:
    raise AssertionError(f"Prediction page failed: {[e.message for e in app.exception]}")
if not app.button:
    raise AssertionError("Prediction button is missing")
print("Streamlit Overview and Predict Activity pages executed without uncaught exceptions.")

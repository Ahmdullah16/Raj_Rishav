# Deployment — recovered CNN–LSTM checkpoint

The public Streamlit app is configured for [rajrishav-har.streamlit.app](https://rajrishav-har.streamlit.app/), repository `Ahmdullah16/Raj_Rishav`, branch `main`, entrypoint `streamlit.py`, Python **3.11**.

## Required assets
`artifacts/best.keras` is a 2.8 MB user-trained Keras model; `artifacts/classes.json` contains its exact seven-label mapping. `run_metadata.json` records verified provenance and an interrupted-training warning. The app loads the model lazily with `st.cache_resource`. Raw videos, tokens and credentials are NOT committed.


## Public HTTP probe observation — browser verification still required (2026-11-08)

GitHub Actions could boot the app, load the exact bundled Keras model, run a
video inference, execute Overview/Prediction pages with AppTest, and obtain
local HTTP 200. **The non-browser probe could not verify anonymous browser access** in the same
probe: `https://rajrishav-har.streamlit.app/_stcore/health` answered HTTP
**303** with repeated redirects to
`https://share.streamlit.io/-/auth/app`. A 303 login/session redirect does **not** establish that the app is private:
Streamlit may establish a browser session using cookies and JavaScript that
plain HTTP clients lack. A real anonymous/incognito browser check is required.
Do not diagnose a sharing-configuration blocker from this HTTP status alone.

**Manual verification by the signed-in Streamlit Community Cloud developer**:

1. Open [Streamlit workspace](https://share.streamlit.io/) and choose the
   workspace owning `Ahmdullah16/Raj_Rishav`.
2. Find `rajrishav-har` → three-dot menu → **Settings** → **Sharing**.
3. Under **Who can view this app**, **confirm** whether **This app is public
   and searchable** is already selected. Change and save **only if private**.
   The app's Share button may also offer **Make this app public**.
4. Confirm repository `Ahmdullah16/Raj_Rishav`, branch `main`,
   entrypoint `streamlit.py`, and Python 3.11.
5. Open the app in a new private/incognito browser window. Check that a
   non-signed-in visitor can reach Predict Activity. Upload a small, authorized
   real MP4 and verify seven model-generated probabilities and a CSV download.
6. If the app itself then fails, use **Manage app → Cloud logs** to diagnose
   package/version or file errors. Do not change the dataset or original
   checkpoint to work around an account-level privacy setting.

Official Streamlit instructions:
https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app

The GitHub workflow's `scripts/probe_public.py` records redirect responses as
**inconclusive** without making a false private/public determination. The
`smoke_streamlit.py` CI test exercises synthetic MP4 and ZIP-wrapped MP4
uploads through the Streamlit AppTest widget and predicts with the pinned
checkpoint. Neither test substitutes for a public browser upload.

## Verify after integration
1. Confirm GitHub Actions tests pass and the app is deployed from the correct `main` commit.
2. Confirm the Overview/Prediction page says a checkpoint exists; no visitor model upload should be necessary.
3. Upload a real supported MP4; record the seven probabilities, predicted label, CPU inference latency and decoding errors.
4. Repeat with a ZIP-wrapped MP4, which the video decoder safely unpacks to temporary local storage.
5. Check Streamlit Community Cloud logs for TensorFlow/Keras version/import problems and memory exhaustion.
6. Do not claim that the partial model has a valid held-out accuracy or production-quality activity recognition.

## Resource and environment guidance
Install `requirements.txt` under Python 3.11; `tensorflow-cpu` and Keras are lazily imported only for prediction. Keep upload cap 150 MB, never cache videos across visitors, and do not re-download the 2 GB dataset. Do not add Google Drive service credentials to GitHub.

To improve predictive quality, resume/finish the Colab experiment and replace the checkpoint only with another **validated** model and the matching mapping, saving older artifacts and updating checksums. Create true `test_metrics.json` and `confusion_matrix.csv` after held-out evaluation.

## Nine-epoch candidate rollout

A newly recovered nine-epoch checkpoint is retained as
`artifacts/best_epoch9.keras`, with pinned checksum
`56dd0103de1273fd31f22c1c0146ef9abf3dbef5998608cca1c3ff30b668fd1a`.
The prior `artifacts/best.keras` remains available and unchanged.

Streamlit's `resolve_checkpoint()` prefers the newer checksum-verified
archive and falls back to the original if necessary. The Prediction page is
the default landing page. The Results page reads the matching nine-epoch
notebook-console history; it does not display manufactured test metrics.

**Important:** This is an experimental early-training model. The best observed
validation score does not establish test accuracy. The public browser state remains unverified; an HTTP authentication redirect
alone cannot establish whether an account-level sharing change is necessary.

## November 8 regression-test follow-up

CI now exercises the **actual Streamlit file-uploader and prediction button**
for a generated MP4 and a ZIP-wrapped MP4 through Streamlit AppTest. This
verifies the upload-to-decoder-to-model-to-UI path on GitHub Actions runners.
It is synthetic input, not a held-out test set or an external-browser test.
The signed-in Streamlit developer must separately verify the deployed app
from an incognito window using an authorized short real-world video.

## Anonymous Chromium browser diagnostic (November 8 continuation)

The separate GitHub Actions workflow `.github/workflows/public-browser-diagnostic.yml`
launches a **fresh, signed-out Chromium browser** against the actual Community Cloud
URL. If the prediction uploader becomes visible, it uploads a tiny generated
MP4, presses **Analyze video and predict activity**, and checks for the
predicted-activity message plus the CSV download control.

Read the workflow's **har-anonymous-browser-diagnostic** artifact:
`public_browser_diagnostic.json` and a screenshot. A `verified: true`
result proves only this synthetic browser upload path on the deployed site,
**not** scientific recognition accuracy or recognition of a real person.
An authentication interstitial, sleeping app, or timeout is reported
without guessing whether Streamlit sharing is private or whether inference code
is faulty. The diagnostic intentionally does not fail source CI on an external
service condition. It does not collect credentials or the Kaggle dataset.

If the browser probe cannot render the prediction page, inspect **Settings →
Sharing** in the signed-in Streamlit workspace, verify the deployed entrypoint
and recent revision, and review Community Cloud logs. Account sharing is not
controllable from the GitHub repository. After synthetic browser success,
repeat a permitted real-video upload manually as the final public-site check.

# Deployment — recovered CNN–LSTM checkpoint

The public Streamlit app is configured for [rajrishav-har.streamlit.app](https://rajrishav-har.streamlit.app/), repository `Ahmdullah16/Raj_Rishav`, branch `main`, entrypoint `streamlit.py`, Python **3.11**.

## Required assets
`artifacts/best.keras` is a 2.8 MB user-trained Keras model; `artifacts/classes.json` contains its exact seven-label mapping. `run_metadata.json` records verified provenance and an interrupted-training warning. The app loads the model lazily with `st.cache_resource`. Raw videos, tokens and credentials are NOT committed.


## Public access blocker observed 2026-11-08

GitHub Actions could boot the app, load the exact bundled Keras model, run a
video inference, execute Overview/Prediction pages with AppTest, and obtain
local HTTP 200. **The public URL is not anonymously accessible** in the same
probe: `https://rajrishav-har.streamlit.app/_stcore/health` answered HTTP
**303** with repeated redirects to
`https://share.streamlit.io/-/auth/app`. A login redirect is evidence of
a **Streamlit account-level access restriction**, not proof that app code fails.

**Required action by the signed-in Streamlit Community Cloud developer**:

1. Open [Streamlit workspace](https://share.streamlit.io/) and choose the
   workspace owning `Ahmdullah16/Raj_Rishav`.
2. Find `rajrishav-har` → three-dot menu → **Settings** → **Sharing**.
3. Under **Who can view this app**, select **This app is public and searchable**
   and save. (Or use the running app's Share → Make this app public.)
4. Confirm repository `Ahmdullah16/Raj_Rishav`, branch `main`,
   entrypoint `streamlit.py`, and Python 3.11.
5. Reopen the app in a new private/incognito browser window. The Overview and
   Predict Activity pages must load without Streamlit sign-in. Upload a small,
   lawful test video and verify seven model-generated probabilities.
6. If the app itself then fails, use **Manage app → Cloud logs** to diagnose
   package/version or file errors. Do not change the dataset or original
   checkpoint to work around an account-level privacy setting.

Official Streamlit instructions:
https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app

The GitHub workflow's `scripts/probe_public.py` logs HTTP status without
marking a private external app as a passing browser upload test.

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
validation score does not establish test accuracy. An account-level public
sharing change is still needed if the app URL redirects to authentication.

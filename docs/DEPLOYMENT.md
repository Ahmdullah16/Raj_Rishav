# Deployment — recovered CNN–LSTM checkpoint

The public Streamlit app is configured for [rajrishav-har.streamlit.app](https://rajrishav-har.streamlit.app/), repository `Ahmdullah16/Raj_Rishav`, branch `main`, entrypoint `streamlit.py`, Python **3.11**.

## Required assets
`artifacts/best.keras` is a 2.8 MB user-trained Keras model; `artifacts/classes.json` contains its exact seven-label mapping. `run_metadata.json` records verified provenance and an interrupted-training warning. The app loads the model lazily with `st.cache_resource`. Raw videos, tokens and credentials are NOT committed.

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

# Recovered model assets (2026-11-08)

This folder **contains real saved model artifacts** recovered from the user's
Google Drive after an interrupted Colab run:

- `best.keras`: original CNN–LSTM 3 checkpoint, **2,774,833 bytes**.
- `classes.json`: exact alphabetical order of the seven trained labels.
- `run_metadata.json`: checkpoint SHA-256, architecture and provenance.
- `history.csv`: **only three rounded epoch values reconstructed from the Colab console**; it is not the original CSVLogger output.

A local Keras model-load and real-video inference smoke succeeded, and the
GitHub full-dependency CI now checks TensorFlow/Keras checkpoint loading,
synthetic video decoding, and Streamlit startup. This proves the pipeline can
execute; **it does not establish held-out accuracy**.

**Never overwrite the original checkpoint with an incomplete resumed run.**
Use `notebooks/HAR_Resume_Colab.ipynb` or `python -m har.resume` to continue
from epoch 3 into a separate Google Drive results folder.

After selecting and completing the final trained model, publish only authentic:

- `history.csv` — new uninterrupted/resume-run training history.
- `run_metadata.json` and `training_state.json` — training provenance and progress.
- `test_metrics.json` — single final held-out evaluation.
- `confusion_matrix.csv` — measured seven-class confusion matrix.
- `model_comparison.csv` — only after comparable runs.

Raw dataset footage must stay in Google Drive. Any replacement of the deployed
checkpoint must preserve a backup, matching `classes.json`, and tested
deployment artifact provenance.

## Versioned checkpoint candidate

- `best.keras`: original three-epoch model, 36 Adam updates — preserved.
- `best_epoch9.keras`: new snapshot, nine complete epochs and 108 Adam updates.
- `history_epoch9.csv`: rounded training and validation figures recovered from
  executed notebook output, **not** an original Keras CSVLogger file.
- `run_metadata_epoch9.json`: provenance, SHA-256, class mapping and limitations.

Only the newer version is preferred for deployed inference, and only after
the CI workflow verifies its checksum and an actual TensorFlow forward pass.
Neither version has authentic held-out test accuracy. Do not copy 35% validation
accuracy into a test-result field.

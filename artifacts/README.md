# Optional trained model and metric artifacts

**This directory intentionally has no invented checkpoint or result.**

After training the notebook on real data, place these files here or in `outputs/`:

- `best.keras` — trained Keras model (must match the seven-class dataset).
- `classes.json` — JSON array of labels in the **exact order used for training**.
- `history.csv` — Keras CSVLogger history (epoch, accuracy, val_accuracy, loss, val_loss).
- `run_metadata.json` — training provenance.
- `test_metrics.json` — final held-out test evaluation.
- `confusion_matrix.csv` — labeled confusion matrix.
- `model_comparison.csv` — optional `model,test_accuracy,...` table from comparable runs.

Avoid committing large or private `.keras` files. On the deployed Prediction page,
upload a trained checkpoint and its matching `classes.json` as an alternative.
No inference occurs without both. The public app does not download raw 2 GB videos.

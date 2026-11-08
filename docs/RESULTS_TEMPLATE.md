# Results log (to fill from real execution)

**Status**: Source code, verified video metadata and splits are prepared. **No new Kaggle-subset CNN–LSTM training run is claimed here.**

| Experiment | Epochs completed | Best validation accuracy | Final held-out test accuracy | Macro F1 | Evidence |
|---|---:|---:|---:|---:|---|
| CNN–LSTM 1 | — | — | — | — | `outputs/variant_1/` |
| CNN–LSTM 2 | — | — | — | — | `outputs/variant_2/` |
| CNN–LSTM 3 | — | — | — | — | `outputs/variant_3/` |
| MobileNetV2–LSTM (optional) | — | — | — | — | `outputs/mobilenetv2/` |

Populate only after executed training. Store `.keras` checkpoints, `history.csv`, `test_metrics.json`, confusion matrix, training plots and runtime configuration. Keep the test partition closed until the final chosen experiment.

Prior *UCF101* result stated by the supplied PDF: Model 3 had 15.42% on a reused validation/evaluation subset. That **cannot be transferred** to the present Kaggle seven-class dataset.

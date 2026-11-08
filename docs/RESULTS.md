# Experiment results — recovery snapshot (2026-11-08)

The following are **real values recovered from the executed Colab notebook's training output**, rounded to four decimals in the notebook's console log. They are not a complete training experiment.

| Completed epoch | Training accuracy | Training loss | Validation accuracy | Validation loss |
|---:|---:|---:|---:|---:|
| 1 | 0.1087 | 1.9703 | 0.1500 | 1.9452 |
| 2 | 0.1957 | 1.9446 | 0.1500 | 1.9433 |
| 3 | 0.1304 | 1.9461 | 0.1500 | 1.9418 |

Epoch 4 began but did not complete. The checkpoint was saved in the original Drive folder at 2026-11-08 10:41:18 (Keras ZIP metadata; timezone unspecified). It reloads and produces seven probabilities for a decoded real video, but the model is **not yet accurate enough for dependable activity recognition**.

**Functional diagnostic only:** The real file `Walking (132).mp4` (a ZIP-wrapped MP4, 110 decoded frames) gave an output tensor `(20,64,64,3)`. Its top prediction was `meet_and_split` (approximately 15.33% probability), not the file's `walking` label. This single test clip is from the held-out split; its result must not be used to tune the model or reported as held-out test accuracy.

**Unavailable:** Complete history, final held-out test accuracy, precision/recall/F1, confusion matrix, other-model comparisons. No estimates are substituted for measurements.

Source: [executed Colab notebook](https://drive.google.com/file/d/1MT3-qjagwKyc8w3beMbFu60OInMKk_O5/view), cells 15 and 17; recovered checkpoint `HAR_Results/cnn_lstm_3/best.keras`.

## Newer recovered snapshot — nine completed epochs (November 8, 2026)

The linked Colab notebook was updated after the original recovery. Its executed
console now shows **nine complete epochs** and a partially executed tenth.
The later Google Drive `best.keras` checkpoint has **108 saved Adam updates**
(9 × 12 batches for 92 training videos, batch size 8). We versioned this
snapshot as `artifacts/best_epoch9.keras`; original `artifacts/best.keras`
remains unchanged.

| Epoch | Training accuracy | Training loss | Validation accuracy | Validation loss |
|---:|---:|---:|---:|---:|
| 1 | 0.1087 | 1.9703 | 0.1500 | 1.9452 |
| 2 | 0.1957 | 1.9446 | 0.1500 | 1.9433 |
| 3 | 0.1304 | 1.9461 | 0.1500 | 1.9418 |
| 4 | 0.1413 | 1.9467 | 0.1500 | 1.9356 |
| 5 | 0.0978 | 1.9431 | 0.2000 | 1.9338 |
| 6 | 0.1630 | 1.9317 | 0.1000 | 1.9116 |
| 7 | 0.1957 | 1.9253 | 0.2500 | 1.8713 |
| 8 | 0.2065 | 1.9478 | 0.3500 | 1.8548 |
| 9 | 0.2174 | 1.8597 | 0.2500 | 1.8029 |

These are rounded console numbers, **not** the original Keras CSVLogger output.
The highest reported validation accuracy was 35% on the 20-video validation
partition at epoch 8; validation loss was lowest at epoch 9.
**There is still no held-out test accuracy, F1 or confusion matrix.**

The nine-epoch archive passed a local Keras 3.13.2/Torch backend load and
forward pass (20×64×64×3 input, seven finite softmax probabilities summing to 1).
That demonstrates technical compatibility only; no public deployed upload or
scientific test accuracy is inferred from it.

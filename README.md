# Human Activity Recognition from Video Using CNN and LSTM

**Student project:** Raj Rishav · Department of Electronics and Electrical Engineering · IIT Guwahati  
**GitHub:** https://github.com/Ahmdullah16/Raj_Rishav


## Current public deployment and scientific status (November 8, 2026)

- **Public app:** [rajrishav-har.streamlit.app](https://rajrishav-har.streamlit.app/) is verified by a **signed-out Chromium browser**. The [GitHub Actions public browser test](https://github.com/Ahmdullah16/Raj_Rishav/actions/runs/42491234697) uploaded a tiny generated MP4, obtained a CNN–LSTM predicted activity, and confirmed the completed results screen includes the CSV download control. This is functional evidence, **not** a recognition-accuracy test.
- **Published model:** `artifacts/best_epoch9.keras` is a checksum-pinned, nine-epoch training snapshot; `artifacts/best.keras` is preserved as a rollback. It has 225,735 parameters and seven outputs for 20 RGB frames at 64 × 64 pixels. Uploaded ZIP-wrapped videos use per-request temporary extraction, cleaned after prediction.
- **Training status:** The latest inspected source Drive checkpoint has nine saved epoch-equivalents of Adam optimizer state (108 updates), though the saved original Colab output lists twelve complete epoch reports and part of epoch thirteen (as observed in the 2026-11-08 14:46 UTC notebook snapshot). Do **not** infer that the later two epochs' optimizer states were saved. The highest console-reported validation accuracy was 0.35 on 20 validation clips; **held-out test accuracy is not available**.
- **Next scientific step:** Preserve authenticated Colab runtime or a newer valid backup if present, otherwise use the dedicated [safe resume notebook](https://colab.research.google.com/github/Ahmdullah16/Raj_Rishav/blob/main/notebooks/HAR_Resume_Colab.ipynb) with the existing Drive dataset. Select the best model using validation, then evaluate the frozen 20-video test set once and publish the actual confusion matrix/report. Do not redownload the dataset or retrain from zero.


This repository implements a reproducible, end-to-end video action classifier using **TimeDistributed CNN** spatial features and an **LSTM** temporal model. It adapts the methodology in the supplied `ML_Lab_project.pdf` and earlier UCF101-based notebook to a **real seven-class Kaggle video subset** stored in Google Drive. It includes index/split creation, frame preprocessing, four model variants, memory-bounded training, evaluation, prediction, unit tests and a self-contained Colab notebook.

[**Open complete project in Google Colab**](https://colab.research.google.com/github/Ahmdullah16/Raj_Rishav/blob/main/notebooks/HAR_CNN_LSTM_Complete.ipynb)

## Dataset: observed contents (not synthetic data)

Source: https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset  
Drive: https://drive.google.com/drive/folders/1EIboVuDhhsE_qL27u3u6QrzNyuAhWfbD

| Category | Videos |
|---|---:|
| Clapping | 20 |
| Meet and Split | 19 |
| Sitting | 19 |
| Standing Still | 18 |
| Walking | 19 |
| Walking While Reading Book | 19 |
| Walking While Using Phone | 18 |
| **Total** | **132** |

The videos total **1,862,222,433 bytes (1.862 GB)** in `RAJ Rishav/HAR_Kaggle_Subset/Human Activity Recognition - Video Dataset/<activity>/`. They are **not committed to GitHub**, avoiding oversized binary pushes and redistribution concerns.

### Requested 70/15/15 split

`data/train_val_test_splits_70_15_15.csv` contains a randomly shuffled, stratified, seeded split by source video, without copying the video files:

| Split | Videos | Proportion |
|---|---:|---:|
| Train | 92 | 69.70% |
| Validation | 20 | 15.15% |
| Test | 20 | 15.15% |

Integer rounding makes these the closest possible whole-video totals to a 70/15/15 split for 132 videos. Every category appears in every partition; file paths are unique. Class group IDs/people identities are not available, so actor/scene leakage **cannot be ruled out**.

## Quick start in Colab

1. Open the **Complete Project** notebook from the link above.
2. Choose **Runtime → Change runtime type → GPU** if available.
3. Mount Google Drive. Confirm you see `MyDrive/RAJ Rishav/HAR_Kaggle_Subset` and its seven subfolders.
4. Run the notebook top-to-bottom. It checks every video path, creates/reuses the split, trains CNN–LSTM 3, evaluates on the withheld test set, generates plots, and saves trained checkpoints and class-name mappings to your Drive.
5. Upload another `.mp4` to run an interactive prediction.

The notebook is self-contained; it does not require downloading the Kaggle dataset again and does not assume `tensorflow` is available outside Colab.

## Resume the interrupted November 8 CNN–LSTM 3 training (recommended)

**Do not run the old fresh-training section to continue the checkpoint.**
The original saved `best.keras` already contains the CNN–LSTM weights and a
compiled Adam optimizer with 36 iterations. The original GitHub copy completed three epochs, but the linked Drive checkpoint
may have advanced further. The runner inspects its saved optimizer updates and
refuses to repeat or guess at incomplete epochs. Test metrics remain unavailable.

**[Open the dedicated resume notebook in Colab](https://colab.research.google.com/github/Ahmdullah16/Raj_Rishav/blob/main/notebooks/HAR_Resume_Colab.ipynb)**

This notebook mounts the existing Drive folder, leaves the 1.862 GB dataset
where it already lives, verifies the original checkpoint and exact seven-class
order, then invokes `har.resume` using the automatically detected completed epoch
count and a **new** persistent results directory
(`HAR_Results/cnn_lstm_3_resumed_v1`). Its local 256 MB cache stores sampled
RGB tensors to avoid repeatedly decoding Google Drive videos. It checkpoints
training progress at each completed epoch and preserves the original source
checkpoint. On restart, a compatible backup must exist for a partially
completed continuation.

Run from an environment with TensorFlow and access to the existing mounted Drive:

```bash
PYTHONPATH=src python -m har.resume \
  --data-root '/content/drive/MyDrive/RAJ Rishav/HAR_Kaggle_Subset' \
  --splits data/train_val_test_splits_70_15_15.csv \
  --resume-from '/content/drive/MyDrive/RAJ Rishav/HAR_Results/cnn_lstm_3/best.keras' \
  --class-map artifacts/classes.json \
  --epochs 25 --batch-size 8 \
  --output-dir '/content/drive/MyDrive/RAJ Rishav/HAR_Results/cnn_lstm_3_resumed_v1' \
  --cache-dir /content/har_preprocessed --max-cache-mb 256
```


The original GitHub checkpoint records 36 Adam updates (three 12-batch epochs).
The checkpoint observed in Drive at 2026-11-08 11:33:34 records **60** updates
(five 12-batch epochs), and has a different SHA-256 digest. Do **not** assume
`--initial-epoch 3` for that updated file. The runner automatically checks
optimizer iterations against the frozen 92-video train split and batch size;
if progress is not an exact epoch multiple, it stops instead of guessing.

Only **after** final model selection based on validation results, run this
command with `--evaluate-only` added to obtain a single test evaluation.
The runner refuses to report final metrics or repeat test evaluation unless
there is evidence of a completed fit and no prior `test_metrics.json`.

A separate full-dependency [GitHub Actions smoke workflow](.github/workflows/inference-smoke.yml)
loads the real bundled checkpoint with TensorFlow, predicts from a small
synthetic video, executes the Streamlit Overview and Predict Activity pages,
and tests local HTTP health. These smoke tests **do not** establish final
recognition accuracy or publicly deployed video-upload success.

## Local run

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
python scripts/create_splits.py --manifest data/video_manifest.csv --output data/train_val_test_splits_70_15_15.csv
python -m har.train \
  --data-root '/path/to/RAJ Rishav/HAR_Kaggle_Subset' \
  --splits data/train_val_test_splits_70_15_15.csv \
  --variant cnn_lstm_3 --epochs 25 --batch-size 8 --output-dir outputs
```

The test partition is held out by default. After model selection, execute a **single final run** with `--evaluate-test` to generate the final metrics. `.keras` files and videos stay out of version control.

### Contents

- `notebooks/HAR_CNN_LSTM_Complete.ipynb` — self-contained, commented Google Colab workflow.
- `src/har/data.py` — metadata validation, seven-class labels, reproducible stratified split.
- `src/har/video.py` — uniform 20-frame RGB sampling, resizing and normalization.
- `src/har/model.py` — three CNN–LSTM baselines and frozen MobileNetV2–LSTM.
- `src/har/train.py` — batch-streaming TensorFlow training and optional final test.
- `data/` — actual *video metadata* and deterministic train/val/test CSV; no footage.
- `docs/METHODOLOGY.md` — mathematics, block diagram, protocol, limitations.
- `docs/RESULTS_TEMPLATE.md` — results table that must be filled from executed experiments.
- `tests/` — deterministic split, class-label and leakage checks.

### Source compatibility and honesty

The supplied 3-page PDF and predecessor notebook describe **UCF101, 101 classes**, 20×64×64×3 inputs, and earlier training. The current Kaggle subset has **seven classes**, not UCF101. The PDF's previously reported 15.42% result **is not a result of this new dataset**. No new model accuracy is reported until a real run produces it.

License and access restrictions remain governed by Kaggle and the owners of the recorded videos. See `docs/METHODOLOGY.md` for science and reproducibility notes.

## IMPORTANT: Fixed `.mp4` files that are actually ZIP archives

At least one original file, `Clapping (1).mp4` (9,962,389 bytes), was inspected
and verified to be a ZIP archive containing `Clapping (1).mp4` as a real H.264
video. `cv2.VideoCapture()` on the outer archive reports `isOpened() == False`.
The old notebook therefore failed despite a correct Drive path.

The patched project detects ZIP content using `zipfile.is_zipfile()`, safely
streams the enclosed MP4 into a local Colab runtime cache, verifies each clip
before computing its 70/15/15 split, and processes the unwrapped MP4. It leaves
all Google Drive originals unchanged, so no extra Drive quota is required.
The local cache uses temporary Colab runtime disk; enable sufficient free space
(~2 GB for the observed subset). If preflight encounters unreadable clips, it
logs the failures, excludes them and recomputes all split quotas transparently.

**If you already have the old notebook open in Colab, close its tab and reopen
this new version, then choose Runtime → Restart session → Run all.**

## Streamlit dashboard (`streamlit.py`)

The root entrypoint is **`streamlit.py`**, with the following pages:
Overview, Dataset & Splits, Predict Activity, Results, Compare Models,
Sensitivity & Diagnostics, Concepts & Mathematics, Downloads and About.

```bash
python -m pip install -r requirements.txt
streamlit run streamlit.py
```

On **Streamlit Community Cloud**, choose:

| Setting | Value |
|---|---|
| Repository | `Ahmdullah16/Raj_Rishav` |
| Branch | `main` |
| Entrypoint | `streamlit.py` |
| Python | 3.11 (Advanced settings) |

Deploy at https://share.streamlit.io. Deploying the site is a separate
Streamlit account operation; a GitHub push alone does not start deployment.

The app loads committed metadata (`data/train_val_test_splits_70_15_15.csv`)
without requiring Google Drive access. **Prediction is only enabled with a
trained `.keras` checkpoint and matching `classes.json`**, either uploaded by
a visitor or placed in `artifacts/`/`outputs/`. No results are invented. Model
loading and video decoding run only on demand; ZIP-disguised MP4 files are
unwrapped safely into a temporary cache. Video uploads have a 150 MB limit.

Other page data sources: `history.csv`, `run_metadata.json`, `test_metrics.json`,
`confusion_matrix.csv`, and optional `model_comparison.csv`. Publish these from
real experiments; avoid putting private Kaggle videos or secrets into GitHub.

**Operational caveat:** Streamlit Community Cloud's free runtime is limited;
large TensorFlow checkpoints may exceed available memory. This app is suitable
for the dashboard and small to medium model inference. For large checkpoints,
use a separately hosted inference service and keep the public dashboard as its
frontend.

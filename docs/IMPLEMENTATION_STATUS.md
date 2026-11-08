# Implementation status — 2026-11-08

## Verified source and recovery
- Baseline `main` inspected at `d01ccabd4e7587ab448286944f8804815a191290`.
- Verified the existing Google Drive archive `HAR_Results/cnn_lstm_3/best.keras` (2,774,833 bytes, SHA-256 in `artifacts/run_metadata.json`).
- The Keras archive contains a CNN–LSTM 3 model with input `(None,20,64,64,3)`, seven softmax outputs and **225,735** parameters.
- The training class order was recovered from executed Colab notebook cell 15; it matches `artifacts/classes.json`.
- Three training epochs completed. The fourth was interrupted. The original Drive `history.csv` was empty; the repository CSV reconstructs *rounded notebook console figures*, not the original Keras CSVLogger file.
- The saved checkpoint loaded in Keras 3.13.2 using the Torch backend. One real Drive video, `Walking (132).mp4`, was decoded from a ZIP wrapper and produced a seven-class prediction. It was **misclassified**; this is a functional smoke test, not a performance evaluation.
- The checkpoint is an **early training checkpoint**, not scientifically validated final model. Last three reported validation accuracies were 15% (20 validation clips). Final held-out test metrics do not exist.
- Dataset remains 132 videos with immutable 92 train / 20 validation / 20 test split; no dataset footage is committed.

## Deployment behavior
- The recovered model is bundled at `artifacts/best.keras` with an independently verified `classes.json` so the existing app can load it automatically without visitor checkpoint uploads.
- Streamlit should be set to repository `Ahmdullah16/Raj_Rishav`, branch `main`, entrypoint `streamlit.py` using Python 3.11. Deploy only after merging verified changes.
- External/public Streamlit deployment status must be checked separately; local model inference is not evidence of successful public rollout.

## Pending scientific work
1. Resume/fine-tune on Colab or approved GPU infrastructure from a preserved copy of the checkpoint (optimizer state may differ); do not overwrite the original backup.
2. Run training to completion using the frozen split. Save real `history.csv`, `run_metadata.json`, class labels and best checkpoint.
3. Evaluate the held-out set once after model selection. Export genuine `test_metrics.json`, classification report, confusion matrix and comparison files.
4. Commit or publish updated model artifacts only after validation. Preserve prior model versions and update the provenance and checksum.
5. Run dependency installation / public Streamlit smoke test after the merge, including uploaded MP4 and ZIP-wrapped MP4. Measure latency on deployed hardware.

## Access constraints / next action
The project owner should authorize and launch the Colab training session with a suitable GPU and can then rerun the exact notebook's training/evaluation cells. **Do not claim completed training or public Streamlit validation until those steps actually succeed.**

## Continuation milestone — November 8, 2026 (PR #2)

- Resumed from merged source baseline `cd74eb379739f8cafb6956ba0ce0122f7bd0891e`; no dataset or original checkpoint replaced.
- Keras 3.13.2 inspection of the existing checkpoint confirmed `compile_config` and saved Adam optimizer state with **36 iterations**. This allows stateful continuation of weights and optimizer under a compatible runtime, but it cannot reproduce lost random seeds/data-shuffle state exactly.
- Implemented `scripts/smoke_inference.py`: **full TensorFlow CPU** loads the real checkpoint, decodes a generated short MP4, samples frames through `har.video.sample_video`, and predicts seven finite probabilities. This is **pipeline compatibility evidence**, not test accuracy.
- Implemented `scripts/smoke_streamlit.py` via Streamlit AppTest (Overview and Predict Activity navigation), plus local server HTTP health. This full-dependency workflow has passed for prior incremental PR #2 commits; the final head must also pass before merge.
- Implemented `src/har/cache.py` for a bounded local cache of normalized frames, with source fingerprint checks and no raw dataset download.
- Implemented `src/har/resume.py`, a new training continuation runner. It checks the exact class mapping and split signature, loads the saved optimizer if compatible, requires a **new output directory**, retains per-epoch progress and Keras backup, and blocks repeat held-out evaluation.
- Created `notebooks/HAR_Resume_Colab.ipynb` that mounts the **existing** Drive dataset and resumes `cnn_lstm_3` after epoch 3; it does not call Kaggle and writes into `HAR_Results/cnn_lstm_3_resumed_v1`.
- Added pure Python regression tests for cache integrity, source-checkpoint protection and progress writes. Test fixture corrected after one failed run, then source CI passed.
- Added read-only HTTP health probe for `https://rajrishav-har.streamlit.app/`. This is not a browser/video-upload test.
- Maintained all prior notes about interrupted original training, validation accuracy 15%, and absent held-out test metrics.

### Remaining work and execution boundary

1. Merge PR #2 **only after both source tests and full-dependency smoke pass**.
2. Check the live Community Cloud app revision and upload a real MP4 from a browser. A GitHub Actions local deployment is not a substitute for this.
3. To complete training, run the new Colab notebook in an authorized Colab GPU runtime with access to the existing `MyDrive/RAJ Rishav` folder. This cannot be done by source-code CI and has not been done here.
4. Preserve new training evidence in Drive; select the best model by validation and run held-out evaluation once. Replace public `best.keras` only after final verification and preserve the original model.
5. Compare other model variants only when computational budget permits, without inventing metrics.

**Current model accuracy is not validated. Do not label the final scientific project complete until training, evaluation and public upload inference are evidenced.**

# Methodology: Video Human Activity Recognition with CNN + LSTM

## Source reconciliation

The supplied **ML_Lab_project.pdf** reports earlier *UCF101* experiments (101 classes, 2,010 sequences, 80/20 split and best validation/evaluation accuracy of 15.42% for the third scratch CNN–LSTM). The supplied baseline notebook also targets UCF101, but proposes a three-way split. The **current project instead uses a different Kaggle dataset of seven classes**, downloaded to Google Drive. These experimental datasets and their metrics must never be conflated. The earlier 15.42% score is a historical PDF result, **not a result of training on the new seven-class dataset**.

## Verified Drive subset

- Source: [Human Activity Recognition - Video Dataset](https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset) by sharjeelmazhar.
- Observed in the provided Drive folder: 132 MP4s, seven real folder labels, 1,862,222,433 bytes total (decimal 1.862 GB).
- Input files remain in their source folders; the CSV is the split, not an extra copy.
- Random seed 42. Global quotas 92/20/20 for 132 clips (about 69.70%/15.15%/15.15%). Each class contributes to all three splits.
- Caveat: group/subject IDs are not present in the manifest; random video-level stratification may leak actors, scenes, or near-duplicate recordings. For rigorous generalization claims, group-aware splitting is preferred when grouping metadata becomes available.

## Preprocessing and mathematical representation

For a video with F decoded frames, select T=20 indices

\[i_t=\operatorname{round}\left(\frac{t(F-1)}{T-1}\right),\quad t=0,\ldots,T-1.\]

Read each selected frame with OpenCV, resize to 64x64, convert BGR to RGB, and normalize by 255. Each video becomes a tensor \(X\in[0,1]^{20\times64\times64\times3}\).

## Model

```mermaid
flowchart LR
    A[Video file] --> B[20 uniformly sampled RGB frames]
    B --> C[Resize 64x64 and normalize]
    C --> D[TimeDistributed CNN: 32, 64, 128 filters]
    D --> E[Global average pooling per frame]
    E --> F[LSTM: 128 units]
    F --> G[Dropout: 0.5]
    G --> H[7-way softmax probabilities]
```

Frame-level CNN features are \(z_t=f_\theta(x_t)\). For LSTM input \(z_t\) and prior hidden state \(h_{t-1}\), the gates and cell updates are

\[
\begin{aligned}
 i_t&=\sigma(W_i[z_t,h_{t-1}]+b_i),\quad
 f_t=\sigma(W_f[z_t,h_{t-1}]+b_f),\\
 \tilde c_t&=\tanh(W_c[z_t,h_{t-1}]+b_c),\quad
 c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\\
 o_t&=\sigma(W_o[z_t,h_{t-1}]+b_o),\quad
 h_t=o_t\odot\tanh(c_t).
\end{aligned}
\]

Finally, \(p(y=k\mid X)=\mathrm{softmax}_k(Wh_T+b)\); train with categorical cross-entropy for integer labels:
\[\mathcal L=-\frac1N\sum_{n=1}^N\log p(y_n\mid X_n).\]

Configurations provided: CNN–LSTM 1 (single convolution + flatten + 64-unit LSTM), CNN–LSTM 2 (single convolution + GAP + 64-unit LSTM), CNN–LSTM 3 (three convolutions + GAP + 128-unit LSTM), and frozen ImageNet MobileNetV2 + 128-unit LSTM. Pretrained MobileNetV2 receives input scaled to [-1,1]. **Do not interpret parameter counts from the prior 101-class model as counts for this seven-class version.**

## Scientific protocol

1. Validate and freeze the split manifest before fitting anything.
2. Train on `train`; use `val` for early stopping and model-selection only.
3. Evaluate on `test` **once after** selecting the final configuration.
4. Export accuracy, macro/weighted F1, confusion matrix, per-class precision and recall, and curves. Do not invent results for unexecuted training.
5. Use identical preprocessing at inference time. Store class ordering alongside checkpoints.

## Known limitations

- Only 132 videos, about 18–20 per class; reported scores may be volatile.
- Uniform frame sampling can miss short events; 64×64 inputs lose fine visual detail.
- Colab GPU/runtime and Google Drive I/O affect training speed.
- The dataset license/usage rights should be checked on Kaggle before sharing footage publicly; raw videos are deliberately excluded from this public GitHub repository.

### ZIP disguised as MP4 (verified November 8, 2026)

A specific file on the connected Drive, `Clapping (1).mp4`, was examined. Its
outer file signature is a ZIP archive, with a single real H.264 MP4 inside.
Attempting to open the wrapper directly with OpenCV fails; extracting the inner
video succeeds (307 frames, 1080×1920, approximately 10.23 seconds).

The input path to the mathematical frame sampler is now resolved by
`resolve_video_file(path)`. It passes real video files through and streams
single-member ZIP wrappers to an ephemeral Colab cache. Files are opened only
after validation. Source-video metadata and sample splits are not evidence of
successful decoding; the notebook validates the actual video before training.

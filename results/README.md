# Measured validation results

All values below use the fixed 320-image validation split (IDs 2880-3199).
The blind 400-image test archive has no public ground truth and is never used
to calculate reference metrics.

| Method | PSNR (dB) ↑ | SSIM ↑ | RTX A4000 latency ↓ |
|---|---:|---:|---:|
| Bicubic | 22.8192 ± 0.3689 | 0.5460 ± 0.0204 | 0.10 ms/image* |
| KLA-RestoreNet v1 | 24.7893 ± 0.4068 | 0.6914 ± 0.0176 | 11.56 ms/image |
| KLA-RestoreNet v2 | **26.2962 ± 0.4384** | **0.7004 ± 0.0176** | **11.56 ms/image** |

`±` values are 95% confidence intervals across validation images. Bicubic
latency was measured locally on CPU and is not directly hardware-comparable;
model latency was measured on the allocated NVIDIA RTX A4000 with batch size 8.

KLA-RestoreNet v2 beats bicubic on 319/320 validation images by PSNR and
275/320 by SSIM. Mean gains are +3.4771 dB PSNR and +0.1544 SSIM.

The worst validation case is a stochastic high-frequency texture. This is an
honest failure mode: deterministic 2x super-resolution cannot uniquely recover
random fine-scale content removed during downsampling.

## Qualitative evidence

- [Training curves](../figures/v2_learning_curves.png)
- [Best validation case](../figures/v2_best_003119.png)
- [Median validation case](../figures/v2_median_002994.png)
- [Worst validation case](../figures/v2_worst_002981.png)

Each comparison shows the degraded input, bicubic baseline, KLA-RestoreNet v2,
and clean ground truth. The worst case is retained rather than cherry-picking
only favorable examples.

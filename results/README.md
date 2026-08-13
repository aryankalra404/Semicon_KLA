# Measured validation results

All values below use the fixed 320-image validation split (IDs 2880-3199).
The blind 400-image test archive has no public ground truth and is never used
to calculate reference metrics.

| Method | PSNR (dB) ↑ | SSIM ↑ | RTX A4000 latency ↓ |
|---|---:|---:|---:|
| Bicubic | 22.8192 ± 0.3689 | 0.5460 ± 0.0204 | 0.10 ms/image* |
| KLA-RestoreNet v1 | 24.7893 ± 0.4068 | 0.6914 ± 0.0176 | 11.56 ms/image |
| KLA-RestoreNet v2 | **26.2962 ± 0.4384** | **0.7004 ± 0.0176** | **11.22 ms/image** |

`±` values are 95% confidence intervals across validation images. Bicubic
latency was measured locally on CPU and is not directly hardware-comparable;
v2 latency is warmed batch-1 p50 on the allocated NVIDIA RTX A4000; its p95 is
11.27 ms/image, peak allocated VRAM is 35.63 MiB, and LPIPS is 0.3738 ± 0.0202.
The earlier v1 timing used batch size 8 and is included only as historical
context.

KLA-RestoreNet v2 beats bicubic on 319/320 validation images by PSNR and
275/320 by SSIM. Mean gains are +3.4771 dB PSNR and +0.1544 SSIM.

The worst validation case is a stochastic high-frequency texture. This is an
honest failure mode: deterministic 2x super-resolution cannot uniquely recover
random fine-scale content removed during downsampling.

## Robustness and architecture ablations

The deterministic stress suite applies six degradation scenarios to the fixed
validation targets. Frozen v2 reaches a macro average of **24.9970 dB PSNR**
and **0.6053 SSIM**. The degradation-aware v3 pilot reaches 24.9888 dB and
0.6063 SSIM, but is 1.80x slower at batch 1 and uses more memory. A five-epoch
randomized-order v2 fine-tune reaches 24.9783 dB and 0.6058 SSIM. These are
useful negative results: neither challenger clears the predeclared promotion
gate, so the compact v2 remains the submission model.

| Candidate | Official PSNR | Official SSIM | Stress PSNR | Stress SSIM | Batch-1 p50 |
|---|---:|---:|---:|---:|---:|
| Frozen v2 | **26.2962** | 0.7004 | **24.9970** | 0.6053 | **11.22 ms** |
| Degradation-aware v3 pilot | 26.2943 | **0.7006** | 24.9888 | **0.6063** | 20.13 ms |
| Randomized-order v2, 5 epochs | 26.2886 | 0.7005 | 24.9783 | 0.6058 | not promoted |

Stress inputs are synthetic diagnostics, not a claim about the private test
distribution. They are used to expose sensitivity and compare models under a
fixed seed, never to replace evaluation on official paired validation data.

## Qualitative evidence

- [Training curves](../figures/v2_learning_curves.png)
- [Best validation case](../figures/v2_best_003119.png)
- [Median validation case](../figures/v2_median_002994.png)
- [Worst validation case](../figures/v2_worst_002981.png)

Each comparison shows the degraded input, bicubic baseline, KLA-RestoreNet v2,
and clean ground truth. The worst case is retained rather than cherry-picking
only favorable examples.

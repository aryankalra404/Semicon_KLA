# Measured validation results

## Evidence boundaries

- PSNR/SSIM/LPIPS use the untouched 320-image official paired validation
  partition, never the blind 400-image test inputs.
- The appearance-cluster split from `make_ood_split.py` is an OOD proxy, not a
  claim about undisclosed KLA source identities.
- `evaluate_defects.py` uses controlled synthetic defect-like probes; it does
  not substitute for labeled real semiconductor defects.
- `256→512` is operationally validated, while paired quality metrics currently
  cover the supplied `128→256` data.
- Geometric disagreement ranks risk but is not a calibrated failure
  probability.

All values below use the fixed 320-image validation split (IDs 2880-3199).
The blind 400-image test archive has no public ground truth and is never used
to calculate reference metrics.

| Method | PSNR (dB) ↑ | SSIM ↑ | RTX A4000 latency ↓ |
|---|---:|---:|---:|
| Bicubic | 22.8192 ± 0.3689 | 0.5460 ± 0.0204 | 0.10 ms/image* |
| Gaussian denoise (σ=0.8) + bicubic | 25.5339 | 0.6483 | classical diagnostic |
| Our Model v1 | 24.7893 ± 0.4068 | 0.6914 ± 0.0176 | 11.56 ms/image |
| Our Model v2, frozen | 26.2962 ± 0.4384 | **0.7004 ± 0.0176** | 11.35 ms/image |
| Our Model v2, final fine-tune | **26.3273 ± 0.4409** | **0.7004 ± 0.0176** | **11.34 ms/image** |

`±` values are 95% confidence intervals across validation images. Bicubic
latency was measured locally on CPU and is not directly hardware-comparable;
final latency is warmed batch-1 p50 on the allocated NVIDIA RTX A4000; its p95 is
11.45 ms/image, peak allocated VRAM is 35.70 MiB, and LPIPS is 0.3717 ± 0.0201.
The earlier v1 timing used batch size 8 and is included only as historical
context.

Against bicubic, final mean gains are +3.5082 dB PSNR and +0.1543 SSIM.
Against the stronger Gaussian-denoise + bicubic baseline, final mean gains are
**+0.7934 ± 0.0680 dB PSNR** and **+0.0521 ± 0.0051 SSIM** (paired 95%
confidence intervals). The model wins on **296/320 images by PSNR** and
**297/320 by SSIM**.
Against frozen v2, the fine-tune improves PSNR by 0.0311 dB on 237/320 images
and LPIPS by 0.00209 on 222/320 images. Paired bootstrap intervals exclude zero
for both gains; the SSIM change is statistically indistinguishable from zero.

The worst validation case is a stochastic high-frequency texture. This is an
honest failure mode: deterministic 2x super-resolution cannot uniquely recover
random fine-scale content removed during downsampling.

## Robustness and architecture ablations

The deterministic stress suite applies six degradation scenarios to the fixed
validation targets. Final v2 reaches a macro average of **25.0401 dB PSNR**
and **0.6063 SSIM**. The degradation-aware v3 pilot reaches 24.9888 dB and
0.6063 SSIM, but is 1.80x slower at batch 1 and uses more memory. A five-epoch
randomized-order v2 fine-tune reaches 24.9783 dB and 0.6058 SSIM. These are
useful negative results. A matched v4a range-aware ablation also failed to beat
ordinary v2 fine-tuning, so the compact fine-tuned v2 remains the submission model.

| Candidate | Official PSNR | Official SSIM | Stress PSNR | Stress SSIM | Batch-1 p50 |
|---|---:|---:|---:|---:|---:|
| Final fine-tuned v2 | **26.3273** | 0.7004 | **25.0401** | 0.6063 | **11.34 ms** |
| Frozen v2 | 26.2962 | 0.7004 | 24.9970 | 0.6053 | 11.35 ms |
| Degradation-aware v3 pilot | 26.2943 | **0.7006** | 24.9888 | **0.6063** | 20.13 ms |
| Randomized-order v2, 5 epochs | 26.2886 | 0.7005 | 24.9783 | 0.6058 | not promoted |
| Range-aware v4a matched ablation | 26.3273 | 0.7003 | 25.0244 | 0.6062 | 11.41 ms |

Stress inputs are synthetic diagnostics, not a claim about the private test
distribution. They are used to expose sensitivity and compare models under a
fixed seed, never to replace evaluation on official paired validation data.

## Qualitative evidence

- [Training curves](../figures/v2_learning_curves.png)
- [Representative validation improvement](../figures/presentation_representative.png)
- [Representative result with detail crops](../figures/presentation_representative_detailed.png)
- [Known over-smoothing limitation](../figures/limitation_oversmoothing_002994.png)

The evidence figure is selected deterministically across all 320 validation
pairs. It excludes visually uninformative low-contrast/low-edge targets,
requires coherent directional structure, and requires model PSNR and SSIM to
improve over both bicubic and Gaussian-denoise + bicubic while gradient
fidelity improves over bicubic, excludes the easiest
top 5% by model PSNR, and ranks the remainder by quality gains and target edge
content. It includes full images, an automatically selected detail crop, and
per-method metrics. The foliage case that visibly over-smooths thin stochastic
detail is retained as an explicit limitation rather than hidden.

Generate both figures inside the pinned CUDA container:

```bash
./run_gpu_matrix.sh presentation-figures
```

The exact selection rule and all per-image measurements are written to
`figures/presentation_cases.json` and `figures/validation_case_metrics.csv`.

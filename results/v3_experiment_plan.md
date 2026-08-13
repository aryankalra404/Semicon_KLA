# v3 promotion protocol

The committed v2 model is the frozen control:

- PSNR: 26.296219
- SSIM: 0.700410
- LPIPS: 0.373796
- Parameters: 580,609
- RTX A4000 latency at batch 8: 11.551 ms/image

The paired-data audit in `degradation_statistics.json` shows that the official
inputs often exceed `[0, 1]`: the median maximum is 1.399 and the median fraction
outside the nominal range is 1.20%. The median residual standard deviation after
a fitted gain/offset correction is 0.083. Consequently, v3 never clips training
inputs and samples compound noise over the observed central range.

## Controlled ablations

All variants warm-start from `weights/final.pt`, use the same split/seed, and
retain width 48 and 12 LR blocks.

| Run | Random order | Conditioning | HR blocks | Consistency |
|---|---:|---:|---:|---:|
| A | yes | no | 0 | 0 |
| B | yes | yes | 0 | 0 |
| C | yes | yes | 2 | 0 |
| D (full v3) | yes | yes | 2 | 0.03 |

The full v3 model has 763,641 parameters. A variant is not promoted based only
on the balanced checkpoint score. Report PSNR, SSIM, LPIPS, six-scenario stress
macro average, batch-1 p50/p95 latency, and peak VRAM.

## Promotion gate

Promote v3 only when at least one condition is satisfied:

1. both PSNR and SSIM improve over v2; or
2. LPIPS and stress robustness improve while PSNR loss is at most 0.10 dB and
   SSIM loss is at most 0.002.

The selected model must also remain below 15 ms/image on the target benchmark
hardware or provide a documented quality/latency Pareto advantage.

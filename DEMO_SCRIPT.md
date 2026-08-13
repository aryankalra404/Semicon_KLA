# Five-minute KLA demo script

## 0:00-0:30 — Problem and contract

Show one official `128x128` NoisyLR array and its `256x256` target. Explain that
Gaussian noise, speckle noise, and downsampling may appear together in any
order, and that the hidden set includes both similar and dissimilar cases.

## 0:30-1:15 — Data hygiene

Run `python audit_data.py`. Show that the paired train set is complete, the
fixed validation split is disjoint, raw NoisyLR values outside `[0,1]` are
preserved, and blind test filenames are never mapped to training ground truth.

## 1:15-2:15 — Model

Explain the compact residual restoration trunk and learned 2x PixelShuffle
upsampling. Show the 580,609-parameter count. Emphasize EMA checkpointing,
robust pixel/SSIM/edge loss, deterministic seeds, gradient clipping, and
synthetic robustness evaluation rather than claiming that synthetic data
perfectly matches the hidden distribution.

## 2:15-3:20 — Results and honest failure analysis

Show bicubic versus v2 quantitative results with 95% confidence intervals,
LPIPS, batch-1 p50/p95 latency, and peak VRAM. Then show best, median, and worst
validation cases. State that random high-frequency detail removed by
downsampling is not uniquely recoverable and is the main observed failure mode.

## 3:20-4:10 — Hidden-test robustness

Show the deterministic six-scenario stress suite: downsampling, high Gaussian,
high speckle, blur, and both degradation orders. Show the v3 and randomized-
order experiments as controlled ablations; explain why the faster v2 remains
the frozen default unless a challenger passes predeclared promotion gates.

## 4:10-4:45 — Standalone evaluator

From a clean Docker container, run:

```bash
python inference.py --input-dir /input/NoisyLR --output-dir /output/restored
```

Then run `python submission_audit.py`. Show exactly 400 finite float32 outputs,
original filenames, shape `(256,256)`, range `[0,1]`, and the aggregate output
hash.

## 4:45-5:00 — Close

End with the public GitHub URL and one sentence: the submission prioritizes
measurable restoration quality, hidden-degradation robustness, and a small,
reproducible evaluator that can run as-is on KLA hardware.

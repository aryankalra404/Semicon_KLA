# Semicon KLA Image Restoration

Reproducible AI restoration pipeline for the **SEMICON India Hackathon 2026
KLA challenge**. It maps noisy `128x128` grayscale NumPy arrays to clean
`256x256` outputs while preserving fine semiconductor structures.

## Results

On the fixed 320-image validation split, the compact 580,609-parameter final
model achieves **26.2962 dB PSNR**, **0.7004 SSIM**, and **11.56 ms/image** on
an NVIDIA RTX A4000. It improves over bicubic by +3.4771 dB and +0.1544 SSIM
on average. Full metrics, confidence intervals, and failure analysis are in
[`results/`](results/README.md).

![Validation learning curves](figures/v2_learning_curves.png)

## Dataset

Place the official extracted release in this layout:

```text
data/train/GT/          # 000000.npy ... 003199.npy
data/train/NoisyLR/     # paired training degradations
data/test/NoisyLR/      # 000000.npy ... 000399.npy blind degradations
```

The blind test archive restarts its numbering at zero; its filenames do not map
to equally numbered training targets. Training uses paired IDs 0-2879 and
validation uses 2880-3199. The raw data is intentionally excluded from Git.

## Environment

Python 3.10-3.12 is recommended for CUDA environments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verify the project

```bash
python -m unittest discover -s tests -v
python audit_data.py
```

## Bicubic baseline

```bash
python evaluate.py \
  --method bicubic \
  --split val \
  --input-dir data/train/NoisyLR \
  --gt-dir data/train/GT \
  --json-output outputs/bicubic_metrics.json
```

## Train

Run a one-epoch smoke test before committing GPU time:

```bash
python train.py --epochs 1 --batch-size 4 --workers 0 --width 16 --blocks 2 \
  --limit-train 16 --limit-val 8 --output-dir weights/smoke
```

Full starting configuration:

```bash
python train.py --epochs 30 --batch-size 8 --width 48 --blocks 12 \
  --synthetic-probability 0.2 --output-dir weights/v2
```

The trainer uses exponential moving-average weights and writes separate
`best_psnr.pt`, `best_ssim.pt`, and `best_balanced.pt` checkpoints. The balanced
score is `PSNR + 10*SSIM`.
Interrupted runs can resume without discarding optimizer state:

```bash
python train.py --epochs 50 --batch-size 8 --width 48 --blocks 12 \
  --resume weights/last.pt
```

## Evaluate a trained checkpoint

```bash
python evaluate.py \
  --method model \
  --split val \
  --weights weights/final.pt \
  --input-dir data/train/NoisyLR \
  --gt-dir data/train/GT \
  --json-output outputs/model_metrics.json
```

Add `--lpips --per-image-output outputs/v2_per_image.csv` to compute the
perceptual metric and save case-level results when the optional LPIPS dependency
is installed.

## Standalone inference

This is the competition-facing command. Ground truth is not required.

```bash
python inference.py \
  --input-dir /path/to/NoisyLR \
  --output-dir /path/to/restored
```

Outputs retain the original filenames and are saved as `256x256` float32
`.npy` arrays in `[0, 1]`.

The repository includes the compact inference-only `weights/final.pt` model.
It contains no optimizer state and loads with the same standalone inference
command used by the benchmark team.

## Submission audit

With official test inputs present locally, this command checks the committed
checkpoint hash and parameter count, runs standalone inference in a subprocess,
and validates output filenames, shapes, dtype, finiteness, and range:

```bash
python validate_submission.py --device auto
```

For a reproducible NVIDIA environment:

```bash
docker build -t kla-restorenet .
docker run --rm --gpus all \
  -v "$PWD/data":/workspace/project/data:ro \
  kla-restorenet python validate_submission.py --device cuda
```

# Semicon KLA Image Restoration

Reproducible AI restoration pipeline for the **SEMICON India Hackathon 2026
KLA challenge**. It maps noisy `128x128` grayscale NumPy arrays to clean
`256x256` outputs while preserving fine semiconductor structures.

## Dataset

Place the official extracted release in this layout:

```text
data/train/GT/          # 000000.npy ... 003199.npy
data/train/NoisyLR/     # paired training degradations
data/test/NoisyLR/      # 000000.npy ... 000399.npy blind degradations
```

IDs 0-399 are reserved for final testing. Training uses 400-2879 and validation
uses 2880-3199, preventing the model from seeing test ground-truth structures.
The raw data is intentionally excluded from Git.

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
```

## Bicubic baseline

```bash
python evaluate.py \
  --method bicubic \
  --input-dir data/test/NoisyLR \
  --gt-dir data/train/GT \
  --json-output outputs/bicubic_metrics.json
```

## Train

Run a one-epoch smoke test before committing GPU time:

```bash
python train.py --epochs 1 --batch-size 4 --workers 0 --width 16 --blocks 2
```

Full starting configuration:

```bash
python train.py --epochs 50 --batch-size 8 --width 48 --blocks 12
```

The best validation-SSIM checkpoint is written to `weights/best.pt`.

## Evaluate a trained checkpoint

```bash
python evaluate.py \
  --method model \
  --weights weights/best.pt \
  --input-dir data/test/NoisyLR \
  --gt-dir data/train/GT \
  --json-output outputs/model_metrics.json
```

## Standalone inference

This is the competition-facing command. Ground truth is not required.

```bash
python inference.py \
  --input-dir /path/to/NoisyLR \
  --output-dir /path/to/restored \
  --weights weights/best.pt
```

Outputs retain the original filenames and are saved as `256x256` float32
`.npy` arrays in `[0, 1]`.

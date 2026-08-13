# Reproducible environment

The reference environment is the NVIDIA NGC container:

```text
nvcr.io/nvidia/pytorch:26.07-py3
digest: sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
PyTorch: 2.13.0a0+9186a08b2c.nv26.07
```

The model was trained and benchmarked on an NVIDIA RTX A4000 (16 GB) using
NVIDIA driver 580.173.02. CUDA forward compatibility was enabled by the NGC
container. The Dockerfile pins the base image tag and `requirements.txt` lists
the direct Python dependencies needed outside NGC.

For exact evaluator reproduction, prefer Docker:

```bash
docker build -t kla-restorenet .
docker run --rm --gpus all --ipc=host \
  -v "$PWD/data/test/NoisyLR":/inputs:ro \
  -v "$PWD/outputs/rehearsal":/outputs \
  kla-restorenet python inference.py \
    --input-dir /inputs --output-dir /outputs --device cuda
```

The NGC image and its bundled frameworks remain subject to NVIDIA and upstream
licenses. `LICENSE` covers repository-authored code only.

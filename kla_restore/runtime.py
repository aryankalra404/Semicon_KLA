"""Shared checkpoint and device helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import RestorationModel, build_model


def mps_is_usable() -> bool:
    """Check actual MPS allocation, not only PyTorch's availability flag."""
    if not torch.backends.mps.is_available():
        return False
    try:
        torch.empty(1, device="mps")
    except RuntimeError:
        return False
    return True


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not mps_is_usable():
            raise RuntimeError(
                "MPS was requested but cannot allocate tensors on this macOS; "
                "use --device cpu or train on a CUDA environment"
            )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_is_usable():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights: str | Path, device: torch.device) -> RestorationModel:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain a 'model' state dictionary")
    config = checkpoint.get("model_config", {})
    model = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()

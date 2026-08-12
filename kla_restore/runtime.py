"""Shared checkpoint and device helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import KLARestoreNet


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights: str | Path, device: torch.device) -> KLARestoreNet:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain a 'model' state dictionary")
    config = checkpoint.get("model_config", {})
    model = KLARestoreNet(
        width=int(config.get("width", 48)),
        blocks=int(config.get("blocks", 12)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


"""Deterministic geometric self-ensemble and checkpoint-ensemble inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _apply_transform(image: torch.Tensor, index: int) -> torch.Tensor:
    rotations = index % 4
    transformed = torch.rot90(image, rotations, dims=(-2, -1))
    if index >= 4:
        transformed = transformed.flip(-1)
    return transformed.contiguous()


def _invert_transform(image: torch.Tensor, index: int) -> torch.Tensor:
    transformed = image.flip(-1) if index >= 4 else image
    rotations = index % 4
    if rotations:
        transformed = torch.rot90(transformed, -rotations, dims=(-2, -1))
    return transformed.contiguous()


def transform_indices(mode: str) -> tuple[int, ...]:
    if mode == "x1":
        return (0,)
    if mode == "x4":
        return (0, 1, 2, 3)
    if mode == "x8":
        return tuple(range(8))
    raise ValueError("self_ensemble must be one of: x1, x4, x8")


@torch.inference_mode()
def restore(
    models: Sequence[nn.Module], inputs: torch.Tensor, self_ensemble: str = "x1"
) -> torch.Tensor:
    """Average checkpoint and invertible geometric predictions."""
    if not models:
        raise ValueError("At least one model is required")
    prediction_sum: torch.Tensor | None = None
    predictions = 0
    for index in transform_indices(self_ensemble):
        transformed = _apply_transform(inputs, index)
        for model in models:
            prediction = _invert_transform(model(transformed), index)
            prediction_sum = (
                prediction
                if prediction_sum is None
                else prediction_sum + prediction
            )
            predictions += 1
    assert prediction_sum is not None
    return prediction_sum / predictions

"""Structure-preserving restoration objectives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import ssim


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + eps * eps).mean()


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return charbonnier(pred_x, target_x) + charbonnier(pred_y, target_y)


def low_frequency_data_consistency(
    prediction: torch.Tensor, observation: torch.Tensor
) -> torch.Tensor:
    """Match observable LR structure without forcing the model to copy noise."""
    projected = F.interpolate(
        prediction, size=observation.shape[-2:], mode="area"
    )
    # A local low-pass suppresses Gaussian/speckle noise in the consistency
    # target. The paired HR objective remains authoritative for fine detail.
    projected = F.avg_pool2d(projected, kernel_size=5, stride=1, padding=2)
    observed = F.avg_pool2d(
        observation.clamp(0.0, 1.0), kernel_size=5, stride=1, padding=2
    )
    return charbonnier(projected, observed)


class RestorationLoss(nn.Module):
    def __init__(
        self,
        pixel_weight: float = 0.7,
        ssim_weight: float = 0.2,
        edge_weight: float = 0.1,
        consistency_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.pixel_weight = pixel_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        observation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pixel = charbonnier(prediction, target)
        structural = 1.0 - ssim(prediction, target).mean()
        edge = gradient_loss(prediction, target)
        total = (
            self.pixel_weight * pixel
            + self.ssim_weight * structural
            + self.edge_weight * edge
        )
        consistency = prediction.new_zeros(())
        if self.consistency_weight:
            if observation is None:
                raise ValueError("observation is required when consistency_weight > 0")
            consistency = low_frequency_data_consistency(prediction, observation)
            total = total + self.consistency_weight * consistency
        parts = {
            "pixel": float(pixel.detach()),
            "ssim": float(structural.detach()),
            "edge": float(edge.detach()),
            "consistency": float(consistency.detach()),
        }
        return total, parts

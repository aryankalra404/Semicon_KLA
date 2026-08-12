"""Structure-preserving restoration objectives."""

from __future__ import annotations

import torch
from torch import nn

from .metrics import ssim


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + eps * eps).mean()


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return charbonnier(pred_x, target_x) + charbonnier(pred_y, target_y)


class RestorationLoss(nn.Module):
    def __init__(self, pixel_weight: float = 0.7, ssim_weight: float = 0.2, edge_weight: float = 0.1) -> None:
        super().__init__()
        self.pixel_weight = pixel_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pixel = charbonnier(prediction, target)
        structural = 1.0 - ssim(prediction, target).mean()
        edge = gradient_loss(prediction, target)
        total = (
            self.pixel_weight * pixel
            + self.ssim_weight * structural
            + self.edge_weight * edge
        )
        parts = {
            "pixel": float(pixel.detach()),
            "ssim": float(structural.detach()),
            "edge": float(edge.detach()),
        }
        return total, parts


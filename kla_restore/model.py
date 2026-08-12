"""A compact NAF-style network for joint denoising and 2x restoration."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=1)
        return left * right


class NAFBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, hidden * 2, 1)
        self.depthwise = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2
        )
        self.gate = SimpleGate()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1),
        )
        self.project = nn.Conv2d(hidden, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.ffn_expand = nn.Conv2d(channels, hidden * 2, 1)
        self.ffn_project = nn.Conv2d(hidden, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.depthwise(self.expand(self.norm1(x)))
        features = self.gate(features)
        features = features * self.attention(features)
        x = x + self.project(features) * self.beta

        features = self.gate(self.ffn_expand(self.norm2(x)))
        return x + self.ffn_project(features) * self.gamma


class KLARestoreNet(nn.Module):
    """Restore a single-channel input to twice its spatial resolution."""

    def __init__(self, width: int = 48, blocks: int = 12) -> None:
        super().__init__()
        if width < 8 or blocks < 1:
            raise ValueError("width must be >= 8 and blocks must be >= 1")
        self.width = width
        self.blocks = blocks
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        self.body = nn.Sequential(*(NAFBlock(width) for _ in range(blocks)))
        self.body_tail = nn.Conv2d(width, width, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(width, 4 * width, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected NCHW grayscale input, got {tuple(x.shape)}")
        baseline = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        stem = self.stem(x)
        residual = self.upsample(stem + self.body_tail(self.body(stem)))
        return (baseline + residual).clamp(0.0, 1.0)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


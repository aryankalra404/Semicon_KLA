"""Compact NAF-style networks for blind denoising and 2x restoration."""

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


def range_aware_input(x: torch.Tensor) -> torch.Tensor:
    """Expose KLA speckle excursions without discarding the raw observation.

    Channels are raw intensity, nominal-range intensity, positive overflow, and
    negative overflow. The representation is deterministic and parameter-free.
    """
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected NCHW grayscale input, got {tuple(x.shape)}")
    return torch.cat(
        (
            x,
            x.clamp(0.0, 1.0),
            (x - 1.0).clamp_min(0.0),
            (-x).clamp_min(0.0),
        ),
        dim=1,
    )


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


class DegradationEncoder(nn.Module):
    """Encode image-level degradation evidence without estimating a kernel."""

    def __init__(self, condition_dim: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        # Four robust global statistics complement the learned local evidence.
        self.project = nn.Sequential(
            nn.Linear(32 + 4, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, condition_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.var(dim=(-2, -1), keepdim=True, unbiased=False).add(1e-6).sqrt()
        normalized = ((x - mean) / std).clamp(-6.0, 6.0)
        learned = self.features(normalized).flatten(1)
        grad_x = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(dim=(-2, -1))
        grad_y = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=(-2, -1))
        outside = ((x < 0.0) | (x > 1.0)).float().mean(dim=(-2, -1))
        statistics = torch.cat(
            (mean.flatten(1), std.log().flatten(1), grad_x + grad_y, outside), dim=1
        )
        return self.project(torch.cat((learned, statistics), dim=1))


class ConditionalNAFBlock(nn.Module):
    """NAF block modulated by an implicit degradation representation."""

    def __init__(self, channels: int, condition_dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.modulation1 = nn.Linear(condition_dim, 2 * channels)
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
        self.modulation2 = nn.Linear(condition_dim, 2 * channels)
        self.ffn_expand = nn.Conv2d(channels, hidden * 2, 1)
        self.ffn_project = nn.Conv2d(hidden, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        # Start exactly as an unconditioned NAF block and learn conditioning safely.
        nn.init.zeros_(self.modulation1.weight)
        nn.init.zeros_(self.modulation1.bias)
        nn.init.zeros_(self.modulation2.weight)
        nn.init.zeros_(self.modulation2.bias)

    @staticmethod
    def modulate(
        features: torch.Tensor, projection: nn.Linear, condition: torch.Tensor
    ) -> torch.Tensor:
        scale, shift = projection(condition).chunk(2, dim=1)
        return features * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        normalized = self.modulate(self.norm1(x), self.modulation1, condition)
        features = self.gate(self.depthwise(self.expand(normalized)))
        features = features * self.attention(features)
        x = x + self.project(features) * self.beta

        normalized = self.modulate(self.norm2(x), self.modulation2, condition)
        features = self.gate(self.ffn_expand(normalized))
        return x + self.ffn_project(features) * self.gamma


class FrequencyMultiScaleBranch(nn.Module):
    """Predict an LR feature correction from complementary frequency scales.

    The branch sees deterministic image-frequency evidence alongside the
    validated v2 trunk features. Its final projection is initialized to zero,
    making v4b exactly identical to its v2 warm start before fine-tuning.
    """

    def __init__(self, trunk_width: int, branch_width: int, blocks: int) -> None:
        super().__init__()
        self.signal_stem = nn.Conv2d(4, branch_width, 3, padding=1)
        self.trunk_projection = nn.Conv2d(trunk_width, branch_width, 1)
        self.coarse_projection = nn.Sequential(
            nn.Conv2d(trunk_width, branch_width, 1),
            nn.Conv2d(
                branch_width,
                branch_width,
                3,
                padding=1,
                groups=branch_width,
            ),
        )
        self.body = nn.Sequential(*(NAFBlock(branch_width) for _ in range(blocks)))
        self.project = nn.Conv2d(branch_width, trunk_width, 3, padding=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    @staticmethod
    def frequency_signals(x: torch.Tensor) -> torch.Tensor:
        local = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        broad = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        coarse = F.interpolate(
            F.avg_pool2d(x, kernel_size=4, stride=4),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.cat((x, x - local, x - broad, coarse), dim=1)

    def forward(self, x: torch.Tensor, trunk: torch.Tensor) -> torch.Tensor:
        coarse = F.interpolate(
            F.avg_pool2d(trunk, kernel_size=2, stride=2),
            size=trunk.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = (
            self.signal_stem(self.frequency_signals(x))
            + self.trunk_projection(trunk)
            + self.coarse_projection(coarse)
        )
        return self.project(self.body(features))


class KLARestoreNet(nn.Module):
    """Restore a single-channel input to twice its spatial resolution.

    ``variant="v2"`` preserves the original architecture and state-dict names.
    ``variant="v3"`` adds degradation conditioning and an HR refinement stage.
    ``variant="v4a"`` preserves v2 except for a range-aware four-channel stem.
    ``variant="v4b"`` adds a zero-initialized multi-scale frequency branch.
    """

    def __init__(
        self,
        width: int = 48,
        blocks: int = 12,
        *,
        variant: str = "v2",
        condition_dim: int = 32,
        hr_width: int = 48,
        hr_blocks: int = 2,
        degradation_conditioning: bool = True,
        frequency_width: int = 24,
        frequency_blocks: int = 2,
    ) -> None:
        super().__init__()
        if width < 8 or blocks < 1:
            raise ValueError("width must be >= 8 and blocks must be >= 1")
        if variant not in {"v2", "v3", "v4a", "v4b"}:
            raise ValueError("variant must be 'v2', 'v3', 'v4a', or 'v4b'")
        if variant == "v3" and (condition_dim < 4 or hr_width < 8 or hr_blocks < 0):
            raise ValueError(
                "v3 requires condition_dim >= 4, hr_width >= 8, hr_blocks >= 0"
            )
        if variant == "v4b" and (frequency_width < 8 or frequency_blocks < 1):
            raise ValueError(
                "v4b requires frequency_width >= 8 and frequency_blocks >= 1"
            )
        self.width = width
        self.blocks = blocks
        self.variant = variant
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        if variant == "v4a":
            self.range_stem = nn.Conv2d(3, width, 3, padding=1, bias=False)
            nn.init.zeros_(self.range_stem.weight)
        if variant in {"v2", "v4a", "v4b"}:
            self.body = nn.Sequential(*(NAFBlock(width) for _ in range(blocks)))
            self.body_tail = nn.Conv2d(width, width, 3, padding=1)
            if variant == "v4b":
                self.frequency_width = frequency_width
                self.frequency_blocks = frequency_blocks
                self.frequency_branch = FrequencyMultiScaleBranch(
                    width, frequency_width, frequency_blocks
                )
            self.upsample = nn.Sequential(
                nn.Conv2d(width, 4 * width, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(width, 1, 3, padding=1),
            )
            nn.init.zeros_(self.upsample[-1].weight)
            nn.init.zeros_(self.upsample[-1].bias)
        else:
            self.condition_dim = condition_dim
            self.hr_width = hr_width
            self.hr_blocks = hr_blocks
            self.degradation_conditioning = degradation_conditioning
            if degradation_conditioning:
                self.degradation_encoder = DegradationEncoder(condition_dim)
                self.body = nn.ModuleList(
                    ConditionalNAFBlock(width, condition_dim) for _ in range(blocks)
                )
            else:
                self.body = nn.ModuleList(NAFBlock(width) for _ in range(blocks))
            self.body_tail = nn.Conv2d(width, width, 3, padding=1)
            self.upsample_features = nn.Sequential(
                nn.Conv2d(width, 4 * hr_width, 3, padding=1),
                nn.PixelShuffle(2),
            )
            if degradation_conditioning:
                self.hr_body = nn.ModuleList(
                    ConditionalNAFBlock(hr_width, condition_dim) for _ in range(hr_blocks)
                )
            else:
                self.hr_body = nn.ModuleList(
                    NAFBlock(hr_width) for _ in range(hr_blocks)
                )
            self.output = nn.Conv2d(hr_width, 1, 3, padding=1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected NCHW grayscale input, got {tuple(x.shape)}")
        baseline = F.interpolate(
            x, scale_factor=2, mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        stem = self.stem(x)
        if self.variant == "v4a":
            stem = stem + self.range_stem(range_aware_input(x)[:, 1:])
        if self.variant in {"v2", "v4a", "v4b"}:
            features = stem + self.body_tail(self.body(stem))
            if self.variant == "v4b":
                features = features + self.frequency_branch(x, features)
            residual = self.upsample(features)
            return baseline + residual

        condition = (
            self.degradation_encoder(x) if self.degradation_conditioning else None
        )
        features = stem
        for block in self.body:
            features = (
                block(features, condition) if condition is not None else block(features)
            )
        features = self.upsample_features(stem + self.body_tail(features))
        for block in self.hr_body:
            features = (
                block(features, condition) if condition is not None else block(features)
            )
        residual = self.output(features)
        return baseline + residual


def model_config(model: KLARestoreNet) -> dict[str, int | str | bool]:
    config: dict[str, int | str | bool] = {
        "variant": model.variant,
        "width": model.width,
        "blocks": model.blocks,
    }
    if model.variant == "v3":
        config.update(
            condition_dim=model.condition_dim,
            hr_width=model.hr_width,
            hr_blocks=model.hr_blocks,
            degradation_conditioning=model.degradation_conditioning,
        )
    if model.variant == "v4b":
        config.update(
            frequency_width=model.frequency_width,
            frequency_blocks=model.frequency_blocks,
        )
    return config


def build_model(config: dict[str, object] | None = None) -> KLARestoreNet:
    """Build a model from a legacy v2 or complete experimental config."""
    config = config or {}
    variant = str(config.get("variant", "v2"))
    return KLARestoreNet(
        width=int(config.get("width", 48)),
        blocks=int(config.get("blocks", 12)),
        variant=variant,
        condition_dim=int(config.get("condition_dim", 32)),
        hr_width=int(config.get("hr_width", 48)),
        hr_blocks=int(config.get("hr_blocks", 2)),
        degradation_conditioning=bool(config.get("degradation_conditioning", True)),
        frequency_width=int(config.get("frequency_width", 24)),
        frequency_blocks=int(config.get("frequency_blocks", 2)),
    )


def initialize_v3_from_v2(
    target: KLARestoreNet, source_state: dict[str, torch.Tensor]
) -> tuple[int, int]:
    """Warm-start v3 from v2, including its residual upsampler when compatible."""
    if target.variant != "v3":
        raise ValueError("Warm-start target must be a v3 model")
    target_state = target.state_dict()
    aliases = {
        "upsample.0.weight": "upsample_features.0.weight",
        "upsample.0.bias": "upsample_features.0.bias",
        "upsample.2.weight": "output.weight",
        "upsample.2.bias": "output.bias",
    }
    copied = 0
    copied_parameters = 0
    for source_name, value in source_state.items():
        target_name = aliases.get(source_name, source_name)
        if target_name in target_state and target_state[target_name].shape == value.shape:
            target_state[target_name] = value
            copied += 1
            copied_parameters += value.numel()
    target.load_state_dict(target_state, strict=True)
    return copied, copied_parameters


def initialize_v4a_from_v2(
    target: KLARestoreNet, source_state: dict[str, torch.Tensor]
) -> tuple[int, int]:
    """Warm-start v4a exactly from v2 while zeroing auxiliary stem channels."""
    if target.variant != "v4a":
        raise ValueError("Warm-start target must be a v4a model")
    target_state = target.state_dict()
    target_state["range_stem.weight"].zero_()

    copied = 0
    copied_parameters = 0
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name] = value
        else:
            raise ValueError(f"Cannot transfer v2 tensor {name}")
        copied += 1
        copied_parameters += value.numel()
    target.load_state_dict(target_state, strict=True)
    return copied, copied_parameters


def initialize_v4b_from_v2(
    target: KLARestoreNet, source_state: dict[str, torch.Tensor]
) -> tuple[int, int]:
    """Warm-start v4b exactly from v2 with a zero-output frequency branch."""
    if target.variant != "v4b":
        raise ValueError("Warm-start target must be a v4b model")
    target_state = target.state_dict()
    target_state["frequency_branch.project.weight"].zero_()
    target_state["frequency_branch.project.bias"].zero_()

    copied = 0
    copied_parameters = 0
    for name, value in source_state.items():
        if name not in target_state or target_state[name].shape != value.shape:
            raise ValueError(f"Cannot transfer v2 tensor {name}")
        target_state[name] = value
        copied += 1
        copied_parameters += value.numel()
    target.load_state_dict(target_state, strict=True)
    return copied, copied_parameters


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

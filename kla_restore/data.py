"""Dataset discovery and loading for paired KLA NumPy arrays."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


SPLIT_RANGES = {
    "train": range(0, 2880),
    "val": range(2880, 3200),
}


def deterministic_split_names(
    seed: int, val_fraction: float = 0.1
) -> tuple[list[str], list[str]]:
    """Return a reproducible random split over the 3,200 paired filenames."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    names = [sample_name(index) for index in range(3200)]
    generator = random.Random(seed)
    generator.shuffle(names)
    val_size = round(len(names) * val_fraction)
    return sorted(names[val_size:]), sorted(names[:val_size])


def sample_name(index: int) -> str:
    return f"{index:06d}.npy"


def names_for_split(split: str) -> list[str]:
    """Return the leakage-safe official filenames for a named split."""
    if split not in SPLIT_RANGES:
        choices = ", ".join(sorted(SPLIT_RANGES))
        raise ValueError(f"Unknown split {split!r}; expected one of: {choices}")
    return [sample_name(index) for index in SPLIT_RANGES[split]]


def all_pair_names() -> list[str]:
    return [sample_name(index) for index in range(3200)]


def validate_pairs(lr_dir: Path, gt_dir: Path, names: Iterable[str]) -> None:
    missing_lr = [name for name in names if not (lr_dir / name).is_file()]
    missing_gt = [name for name in names if not (gt_dir / name).is_file()]
    if missing_lr or missing_gt:
        details = []
        if missing_lr:
            details.append(f"missing {len(missing_lr)} LR files")
        if missing_gt:
            details.append(f"missing {len(missing_gt)} GT files")
        raise FileNotFoundError("Dataset validation failed: " + ", ".join(details))


def audit_pairs(lr_dir: Path, gt_dir: Path, names: Iterable[str]) -> None:
    """Fully read arrays before training so truncated transfers fail immediately."""
    names = list(names)
    validate_pairs(lr_dir, gt_dir, names)
    problems: list[str] = []
    for name in names:
        for label, path, expected_shape in (
            ("LR", lr_dir / name, (128, 128)),
            ("GT", gt_dir / name, (256, 256)),
        ):
            try:
                array = np.load(path, allow_pickle=False)
                if array.shape != expected_shape:
                    problems.append(f"{label} {path}: shape={array.shape}")
                elif array.dtype != np.float32:
                    problems.append(f"{label} {path}: dtype={array.dtype}")
                elif not np.isfinite(array).all():
                    problems.append(f"{label} {path}: contains non-finite values")
            except Exception as error:
                problems.append(f"{label} {path}: {type(error).__name__}: {error}")
    if problems:
        preview = "\n".join(problems[:20])
        raise ValueError(f"Dataset audit found {len(problems)} problem(s):\n{preview}")


def load_npy_tensor(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale array at {path}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Expected numeric array at {path}, got {array.dtype}")
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).unsqueeze(0)


def paired_geometric_augmentation(
    lr: torch.Tensor, gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same lossless geometric transform to LR and GT."""
    if random.random() < 0.5:
        lr, gt = lr.flip(-1), gt.flip(-1)
    if random.random() < 0.5:
        lr, gt = lr.flip(-2), gt.flip(-2)
    rotations = random.randrange(4)
    if rotations:
        lr = torch.rot90(lr, rotations, dims=(-2, -1))
        gt = torch.rot90(gt, rotations, dims=(-2, -1))
    return lr.contiguous(), gt.contiguous()


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = 5
    coordinates = (
        torch.arange(kernel_size, dtype=image.dtype, device=image.device)
        - kernel_size // 2
    )
    kernel = torch.exp(-coordinates.square() / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel).view(1, 1, kernel_size, kernel_size)
    return F.conv2d(F.pad(image, (2, 2, 2, 2), mode="reflect"), window)


def _resize_half(image: torch.Tensor) -> torch.Tensor:
    mode = random.choice(("area", "bilinear", "bicubic"))
    kwargs = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(image, scale_factor=0.5, mode=mode, **kwargs)


def synthetic_compound_degradation(
    gt: torch.Tensor, *, policy: str = "randomized"
) -> torch.Tensor:
    """Create blind 2x LR observations from the three official degradations.

    The randomized policy treats blur/downsampling, additive Gaussian noise,
    and multiplicative speckle as independently gated operations and shuffles
    their order. This includes compound cases as well as the important corner
    cases in which only one degradation is prominent.
    """
    if policy not in {"fixed", "randomized"}:
        raise ValueError("policy must be 'fixed' or 'randomized'")
    image = gt.unsqueeze(0)
    if policy == "fixed":
        if random.random() < 0.5:
            image = _gaussian_blur(image, random.uniform(0.4, 1.2))
        image = _resize_half(image)
        image = image * random.uniform(0.92, 1.08) + random.uniform(-0.02, 0.02)
        image = image + torch.randn_like(image) * random.uniform(0.01, 0.07)
        image = image + image.abs() * torch.randn_like(image) * random.uniform(
            0.02, 0.12
        )
        return image[0].contiguous()

    # Always resize exactly once; independently gate the other official
    # degradations, then shuffle to prevent the model learning one fixed order.
    operations = ["resize"]
    if random.random() < 0.70:
        operations.append("blur")
    if random.random() < 0.85:
        operations.append("gaussian")
    if random.random() < 0.85:
        operations.append("speckle")
    if random.random() < 0.50:
        operations.append("radiometric")
    random.shuffle(operations)

    for operation in operations:
        if operation == "resize":
            image = _resize_half(image)
        elif operation == "blur":
            image = _gaussian_blur(image, random.uniform(0.35, 1.35))
        elif operation == "gaussian":
            image = image + torch.randn_like(image) * random.uniform(0.01, 0.09)
        elif operation == "speckle":
            image = image + image.abs() * torch.randn_like(image) * random.uniform(
                0.02, 0.16
            )
        else:
            image = image * random.uniform(0.92, 1.04) + random.uniform(
                -0.01, 0.045
            )
    return image[0].contiguous()


class PairedNpyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Load filename-paired low-resolution and ground-truth arrays."""

    def __init__(
        self,
        lr_dir: str | Path,
        gt_dir: str | Path,
        names: Sequence[str],
        augment: bool = False,
        synthetic_probability: float = 0.0,
        synthetic_policy: str = "fixed",
        audit: bool = False,
    ) -> None:
        self.lr_dir = Path(lr_dir)
        self.gt_dir = Path(gt_dir)
        self.names = list(names)
        self.augment = augment
        if not 0.0 <= synthetic_probability <= 1.0:
            raise ValueError("synthetic_probability must be in [0, 1]")
        self.synthetic_probability = synthetic_probability
        if synthetic_policy not in {"fixed", "randomized"}:
            raise ValueError("synthetic_policy must be 'fixed' or 'randomized'")
        self.synthetic_policy = synthetic_policy
        validate_pairs(self.lr_dir, self.gt_dir, self.names)
        if audit:
            audit_pairs(self.lr_dir, self.gt_dir, self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        name = self.names[index]
        lr = load_npy_tensor(self.lr_dir / name)
        gt = load_npy_tensor(self.gt_dir / name)
        if gt.shape[-2:] != (lr.shape[-2] * 2, lr.shape[-1] * 2):
            raise ValueError(
                f"Expected 2x spatial pairing for {name}, got {lr.shape} -> {gt.shape}"
            )
        if self.augment:
            lr, gt = paired_geometric_augmentation(lr, gt)
            if random.random() < self.synthetic_probability:
                lr = synthetic_compound_degradation(gt, policy=self.synthetic_policy)
        return lr, gt, name


class UnpairedNpyDataset(Dataset[tuple[torch.Tensor, str]]):
    """Load a directory of blind low-resolution arrays for inference."""

    def __init__(self, input_dir: str | Path) -> None:
        self.input_dir = Path(input_dir)
        self.paths = sorted(self.input_dir.glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No .npy inputs found in {self.input_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        return load_npy_tensor(path), path.name

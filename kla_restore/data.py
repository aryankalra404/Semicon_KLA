"""Dataset discovery and loading for paired KLA NumPy arrays."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_RANGES = {
    "train": range(0, 2880),
    "val": range(2880, 3200),
}


def sample_name(index: int) -> str:
    return f"{index:06d}.npy"


def names_for_split(split: str) -> list[str]:
    """Return the leakage-safe official filenames for a named split."""
    if split not in SPLIT_RANGES:
        choices = ", ".join(sorted(SPLIT_RANGES))
        raise ValueError(f"Unknown split {split!r}; expected one of: {choices}")
    return [sample_name(index) for index in SPLIT_RANGES[split]]


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


class PairedNpyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Load filename-paired low-resolution and ground-truth arrays."""

    def __init__(
        self,
        lr_dir: str | Path,
        gt_dir: str | Path,
        names: Sequence[str],
        augment: bool = False,
    ) -> None:
        self.lr_dir = Path(lr_dir)
        self.gt_dir = Path(gt_dir)
        self.names = list(names)
        self.augment = augment
        validate_pairs(self.lr_dir, self.gt_dir, self.names)

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

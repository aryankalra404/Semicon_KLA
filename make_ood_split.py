#!/usr/bin/env python3
"""Create an appearance-cluster-disjoint OOD proxy split without touching official validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kla_restore.data import load_npy_tensor, names_for_split
from kla_restore.robustness import (
    choose_holdout_cluster,
    deterministic_kmeans,
    image_descriptor,
    standardize_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=Path, default=Path("data/train/GT"))
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--target-size", type=int, default=320)
    parser.add_argument("--holdout-cluster", type=int)
    parser.add_argument("--seed", type=int, default=260813)
    parser.add_argument("--output", type=Path, default=Path("splits/ood_cluster.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = names_for_split("train")
    features = np.stack(
        [image_descriptor(load_npy_tensor(args.gt_dir / name)[0].numpy()) for name in names]
    )
    normalized, mean, scale = standardize_features(features)
    labels, centers = deterministic_kmeans(normalized, args.clusters, seed=args.seed)
    holdout = (
        choose_holdout_cluster(labels, args.target_size)
        if args.holdout_cluster is None
        else args.holdout_cluster
    )
    if not 0 <= holdout < args.clusters:
        raise ValueError("holdout cluster is outside the configured range")
    train_names = [name for name, label in zip(names, labels, strict=True) if label != holdout]
    val_names = [name for name, label in zip(names, labels, strict=True) if label == holdout]
    result = {
        "kind": "appearance_cluster_disjoint_ood_proxy",
        "seed": args.seed,
        "clusters": args.clusters,
        "holdout_cluster": holdout,
        "cluster_sizes": {str(index): int((labels == index).sum()) for index in range(args.clusters)},
        "feature_dimensions": int(features.shape[1]),
        "feature_standardization": {"mean": mean.tolist(), "scale": scale.tolist()},
        "cluster_centers": centers.tolist(),
        "train_names": train_names,
        "val_names": val_names,
        "official_validation_untouched": names_for_split("val"),
        "limitations": "Appearance clusters are an OOD proxy, not authoritative source labels.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"train={len(train_names)} ood_proxy_val={len(val_names)} "
        f"holdout_cluster={holdout} output={args.output}"
    )


if __name__ == "__main__":
    main()

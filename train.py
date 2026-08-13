#!/usr/bin/env python3
"""Train KLARestoreNet on the leakage-safe training split."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from kla_restore.data import (
    PairedNpyDataset,
    all_pair_names,
    deterministic_split_names,
    names_for_split,
)
from kla_restore.losses import RestorationLoss
from kla_restore.metrics import psnr, ssim
from kla_restore.model import (
    KLARestoreNet,
    build_model,
    initialize_v3_from_v2,
    initialize_v4a_from_v2,
    initialize_v4b_from_v2,
    model_config,
    parameter_count,
)
from kla_restore.runtime import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("weights"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="For v4b, train only frequency_branch for the first N epochs",
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        help="v4b backbone LR after staged unfreezing (defaults to --learning-rate)",
    )
    parser.add_argument(
        "--branch-learning-rate",
        type=float,
        help="v4b frequency-branch LR (defaults to --learning-rate)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument(
        "--variant", choices=("v2", "v3", "v4a", "v4b"), default="v2"
    )
    parser.add_argument("--condition-dim", type=int, default=32)
    parser.add_argument("--hr-width", type=int, default=48)
    parser.add_argument("--hr-blocks", type=int, default=2)
    parser.add_argument("--frequency-width", type=int, default=24)
    parser.add_argument("--frequency-blocks", type=int, default=2)
    parser.add_argument("--disable-degradation-conditioning", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-train", type=int, help="Optional smoke-test sample limit")
    parser.add_argument("--limit-val", type=int, help="Optional smoke-test sample limit")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--split-seed",
        type=int,
        help="Use a reproducible random 90/10 split instead of fixed IDs",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train on all 3,200 pairs after configuration selection; validation is disabled",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path, help="Resume model and optimizer state")
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Initialize v2, v3, v4a, or v4b from a v2 checkpoint",
    )
    parser.add_argument("--synthetic-probability", type=float, default=0.0)
    parser.add_argument(
        "--synthetic-policy", choices=("fixed", "randomized"), default="fixed"
    )
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--pixel-weight", type=float, default=0.7)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after N validation epochs without balanced-score improvement; 0 disables",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--preservation-weight",
        type=float,
        default=0.0,
        help="L1 weight against a frozen reference model on the same observation",
    )
    parser.add_argument(
        "--preservation-weights",
        type=Path,
        help="Frozen reference checkpoint used by --preservation-weight",
    )
    parser.add_argument(
        "--collapse-guard-psnr-drop",
        type=float,
        default=0.0,
        help="Abort if validation PSNR falls this far below the frozen reference",
    )
    parser.add_argument("--skip-data-audit", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def v4b_parameter_groups(
    model: KLARestoreNet,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Return disjoint backbone and frequency-branch parameter lists."""
    if model.variant != "v4b":
        raise ValueError("Differential parameter groups require a v4b model")
    backbone, branch = [], []
    for name, parameter in model.named_parameters():
        (branch if name.startswith("frequency_branch.") else backbone).append(parameter)
    if not backbone or not branch:
        raise ValueError("v4b parameter grouping produced an empty group")
    return backbone, branch


def set_v4b_stage(model: KLARestoreNet, *, branch_only: bool) -> None:
    """Freeze or unfreeze v4b's inherited backbone without touching its branch."""
    backbone, branch = v4b_parameter_groups(model)
    for parameter in backbone:
        parameter.requires_grad_(not branch_only)
    for parameter in branch:
        parameter.requires_grad_(True)


def validation_psnr_collapsed(
    observed_psnr: float, reference_psnr: float, allowed_drop: float
) -> bool:
    """Return whether an enabled PSNR guard has been crossed."""
    if allowed_drop < 0:
        raise ValueError("allowed_drop must be non-negative")
    return allowed_drop > 0 and observed_psnr < reference_psnr - allowed_drop


@torch.no_grad()
def validate(model: KLARestoreNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    psnr_values, ssim_values = [], []
    for lr, gt, _ in loader:
        lr, gt = lr.to(device), gt.to(device)
        prediction = model(lr).clamp(0.0, 1.0)
        psnr_values.extend(psnr(prediction, gt).cpu().tolist())
        ssim_values.extend(ssim(prediction, gt).cpu().tolist())
    return {
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }


def main() -> None:
    args = parse_args()
    if args.resume and args.initialize_from:
        raise SystemExit("Use only one of --resume and --initialize-from")
    if args.freeze_backbone_epochs < 0 or args.early_stopping_patience < 0:
        raise SystemExit("freeze epochs and early-stopping patience must be non-negative")
    staged_v4b = args.freeze_backbone_epochs > 0
    if staged_v4b and args.variant != "v4b":
        raise SystemExit("--freeze-backbone-epochs is supported only for v4b")
    if min(
        args.early_stopping_min_delta,
        args.preservation_weight,
        args.collapse_guard_psnr_drop,
    ) < 0:
        raise SystemExit("early-stop, preservation, and collapse values must be non-negative")
    if args.preservation_weight > 0 and args.preservation_weights is None:
        raise SystemExit("--preservation-weight requires --preservation-weights")
    if args.collapse_guard_psnr_drop > 0 and args.preservation_weights is None:
        raise SystemExit("--collapse-guard-psnr-drop requires --preservation-weights")
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_all:
        train_names = all_pair_names()
        val_names: list[str] = []
    elif args.split_seed is not None:
        train_names, val_names = deterministic_split_names(args.split_seed)
    else:
        train_names = names_for_split("train")
        val_names = names_for_split("val")
    if args.limit_train is not None:
        train_names = train_names[: args.limit_train]
    if args.limit_val is not None:
        val_names = val_names[: args.limit_val]
    train_set = PairedNpyDataset(
        args.data_root / "NoisyLR",
        args.data_root / "GT",
        train_names,
        augment=True,
        synthetic_probability=args.synthetic_probability,
        synthetic_policy=args.synthetic_policy,
        audit=not args.skip_data_audit,
    )
    val_set = (
        PairedNpyDataset(
            args.data_root / "NoisyLR",
            args.data_root / "GT",
            val_names,
            audit=not args.skip_data_audit,
        )
        if val_names
        else None
    )
    pin_memory = device.type == "cuda"
    data_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
        generator=data_generator,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=pin_memory,
            persistent_workers=args.workers > 0,
        )
        if val_set is not None
        else None
    )

    model = KLARestoreNet(
        args.width,
        args.blocks,
        variant=args.variant,
        condition_dim=args.condition_dim,
        hr_width=args.hr_width,
        hr_blocks=args.hr_blocks,
        degradation_conditioning=not args.disable_degradation_conditioning,
        frequency_width=args.frequency_width,
        frequency_blocks=args.frequency_blocks,
    ).to(device)
    if args.initialize_from:
        source = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        source_config = source.get("model_config", {})
        if source_config.get("variant", "v2") != "v2":
            raise ValueError("--initialize-from checkpoint must contain a v2 model")
        if int(source_config.get("width", 48)) != args.width or int(
            source_config.get("blocks", 12)
        ) != args.blocks:
            raise ValueError(
                "--initialize-from width/blocks must match the requested model "
                f"configuration; checkpoint={source_config}, "
                f"requested width={args.width}/blocks={args.blocks}"
            )
        if args.variant == "v2":
            model.load_state_dict(source["model"], strict=True)
            copied_tensors = len(source["model"])
            copied_parameters = sum(value.numel() for value in source["model"].values())
        else:
            initializers = {
                "v3": initialize_v3_from_v2,
                "v4a": initialize_v4a_from_v2,
                "v4b": initialize_v4b_from_v2,
            }
            initializer = initializers[args.variant]
            copied_tensors, copied_parameters = initializer(model, source["model"])
        source_parameters = sum(value.numel() for value in source["model"].values())
        if copied_parameters != source_parameters:
            raise ValueError(
                "Warm-start did not copy the complete v2 model: "
                f"copied {copied_parameters:,}/{source_parameters:,} parameters. "
                "For exact v3 transfer, set --hr-width equal to --width."
            )
        print(
            f"initialized_from={args.initialize_from} tensors={copied_tensors} "
            f"parameters={copied_parameters:,}"
        )
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    preservation_model = None
    reference_metrics = None
    if args.preservation_weights is not None:
        reference_checkpoint = torch.load(
            args.preservation_weights, map_location="cpu", weights_only=False
        )
        preservation_model = build_model(reference_checkpoint.get("model_config"))
        preservation_model.load_state_dict(reference_checkpoint["model"], strict=True)
        preservation_model = preservation_model.to(device).eval()
        for parameter in preservation_model.parameters():
            parameter.requires_grad_(False)
    loss_fn = RestorationLoss(
        pixel_weight=args.pixel_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        consistency_weight=args.consistency_weight,
    )
    backbone_lr = args.backbone_learning_rate or args.learning_rate
    branch_lr = args.branch_learning_rate or args.learning_rate
    if staged_v4b:
        backbone_parameters, branch_parameters = v4b_parameter_groups(model)
        optimizer = AdamW(
            [
                {"params": backbone_parameters, "lr": 0.0, "group_name": "backbone"},
                {"params": branch_parameters, "lr": branch_lr, "group_name": "branch"},
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = None
    else:
        optimizer = AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_ssim = -1.0
    best_psnr = -1.0
    best_balanced = -1.0
    history = []
    start_epoch = 1
    epochs_without_improvement = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        expected_config = model_config(model)
        checkpoint_config = model_config(build_model(checkpoint.get("model_config")))
        if checkpoint_config != expected_config:
            raise ValueError(
                f"Checkpoint architecture {checkpoint_config} does not "
                f"match requested architecture {expected_config}"
            )
        model.load_state_dict(checkpoint.get("train_model", checkpoint["model"]))
        ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_ssim = float(checkpoint.get("best_ssim", checkpoint["metrics"]["ssim"]))
        best_psnr = float(checkpoint.get("best_psnr", checkpoint["metrics"]["psnr"]))
        best_balanced = float(
            checkpoint.get(
                "best_balanced",
                checkpoint["metrics"]["psnr"] + 10 * checkpoint["metrics"]["ssim"],
            )
        )
        history_path = args.output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text())
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )

    if preservation_model is not None and val_loader is not None:
        reference_metrics = validate(preservation_model, val_loader, device)
        print(json.dumps({"preservation_reference": reference_metrics}))

    print(
        f"device={device} train={len(train_set)} "
        f"val={len(val_set) if val_set is not None else 0} "
        f"parameters={parameter_count(model):,}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        if staged_v4b:
            branch_only = epoch <= args.freeze_backbone_epochs
            set_v4b_stage(model, branch_only=branch_only)
            fine_tune_epochs = max(1, args.epochs - args.freeze_backbone_epochs)
            if branch_only:
                stage_progress = (epoch - 1) / max(1, args.freeze_backbone_epochs)
                current_backbone_lr = 0.0
            else:
                stage_progress = (epoch - args.freeze_backbone_epochs - 1) / fine_tune_epochs
                current_backbone_lr = backbone_lr
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * stage_progress))
            optimizer.param_groups[0]["lr"] = current_backbone_lr * cosine_factor
            optimizer.param_groups[1]["lr"] = branch_lr * cosine_factor
        model.train()
        started = time.perf_counter()
        running_loss = 0.0
        running_parts: dict[str, float] = {}
        for lr, gt, _ in train_loader:
            lr, gt = lr.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(lr)
                loss, parts = loss_fn(prediction, gt, lr)
                if preservation_model is not None and args.preservation_weight > 0:
                    with torch.no_grad():
                        reference_prediction = preservation_model(lr)
                    preservation = F.l1_loss(prediction, reference_prediction)
                    loss = loss + args.preservation_weight * preservation
                    parts["preservation"] = float(preservation.detach())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema_model.parameters(), model.parameters(), strict=True
                ):
                    ema_parameter.lerp_(parameter, 1.0 - args.ema_decay)
            running_loss += float(loss.detach()) * lr.shape[0]
            for name, value in parts.items():
                running_parts[name] = running_parts.get(name, 0.0) + value * lr.shape[0]
        if scheduler is not None:
            scheduler.step()

        metrics = (
            validate(ema_model, val_loader, device)
            if val_loader is not None
            else {"psnr": float("nan"), "ssim": float("nan")}
        )
        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_set),
            "train_parts": {
                name: value / len(train_set) for name, value in running_parts.items()
            },
            "val_psnr": metrics["psnr"],
            "val_ssim": metrics["ssim"],
            "seconds": time.perf_counter() - started,
            "learning_rates": {
                str(group.get("group_name", index)): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
        }
        history.append(record)
        print(json.dumps(record))

        balanced = metrics["psnr"] + 10 * metrics["ssim"]
        improved_balanced = balanced > best_balanced + args.early_stopping_min_delta
        improved_ssim = metrics["ssim"] > best_ssim
        improved_psnr = metrics["psnr"] > best_psnr
        if improved_ssim:
            best_ssim = metrics["ssim"]
        if improved_psnr:
            best_psnr = metrics["psnr"]
        if improved_balanced:
            best_balanced = balanced
            epochs_without_improvement = 0
        elif staged_v4b and epoch <= args.freeze_backbone_epochs:
            # Branch warm-up is intentionally not an early-stopping trial.
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "model": ema_model.state_dict(),
            "train_model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "model_config": model_config(model),
            "training_config": {
                "synthetic_probability": args.synthetic_probability,
                "synthetic_policy": args.synthetic_policy,
                "consistency_weight": args.consistency_weight,
                "ema_decay": args.ema_decay,
                "gradient_clip": args.gradient_clip,
                "seed": args.seed,
                "data_order_seed": args.seed,
                "split_seed": args.split_seed,
                "train_all": args.train_all,
                "loss_weights": {
                    "pixel": args.pixel_weight,
                    "ssim": args.ssim_weight,
                    "edge": args.edge_weight,
                    "consistency": args.consistency_weight,
                },
                "initialized_from": (
                    str(args.initialize_from) if args.initialize_from else None
                ),
                "freeze_backbone_epochs": args.freeze_backbone_epochs,
                "backbone_learning_rate": backbone_lr,
                "branch_learning_rate": branch_lr,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "preservation_weight": args.preservation_weight,
                "preservation_weights": (
                    str(args.preservation_weights) if args.preservation_weights else None
                ),
                "collapse_guard_psnr_drop": args.collapse_guard_psnr_drop,
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "best_ssim": best_ssim,
            "best_psnr": best_psnr,
            "best_balanced": best_balanced,
            "epochs_without_improvement": epochs_without_improvement,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_loader is None:
            torch.save(checkpoint, args.output_dir / "final_all_data.pt")
            (args.output_dir / "history.json").write_text(
                json.dumps(history, indent=2)
            )
            continue
        if reference_metrics is not None and validation_psnr_collapsed(
            metrics["psnr"],
            reference_metrics["psnr"],
            args.collapse_guard_psnr_drop,
        ):
            failure = {
                "aborted": "validation_psnr_collapse",
                "epoch": epoch,
                "reference_psnr": reference_metrics["psnr"],
                "observed_psnr": metrics["psnr"],
                "allowed_drop": args.collapse_guard_psnr_drop,
            }
            (args.output_dir / "ABORTED.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
            (args.output_dir / "history.json").write_text(
                json.dumps(history, indent=2)
            )
            raise SystemExit(json.dumps(failure))
        if improved_ssim:
            torch.save(checkpoint, args.output_dir / "best_ssim.pt")
        if improved_psnr:
            torch.save(checkpoint, args.output_dir / "best_psnr.pt")
        if improved_balanced:
            torch.save(checkpoint, args.output_dir / "best_balanced.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                json.dumps(
                    {
                        "early_stop": True,
                        "epoch": epoch,
                        "best_balanced": best_balanced,
                        "patience": args.early_stopping_patience,
                    }
                )
            )
            break


if __name__ == "__main__":
    main()

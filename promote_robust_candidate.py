#!/usr/bin/env python3
"""Fail-closed promotion gate for a robustness challenger versus frozen v2."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--control-stress", type=Path, required=True)
    parser.add_argument("--candidate-stress", type=Path, required=True)
    parser.add_argument(
        "--ood-fixed-policy",
        type=Path,
        required=True,
        help="OOD score for the matched fixed-policy cluster-split model",
    )
    parser.add_argument(
        "--ood-randomized-policy",
        type=Path,
        required=True,
        help="OOD score for the matched randomized-policy cluster-split model",
    )
    parser.add_argument("--control-defects", type=Path, required=True)
    parser.add_argument("--candidate-defects", type=Path, required=True)
    parser.add_argument("--control-benchmark", type=Path, required=True)
    parser.add_argument("--candidate-benchmark", type=Path, required=True)
    parser.add_argument(
        "--uncertainty-validation",
        type=Path,
        required=True,
        help="Validation evidence that disagreement ranks reconstruction error",
    )
    parser.add_argument("--candidate-weights", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("results/robust_promotion.json"))
    parser.add_argument("--promote-to", type=Path)
    parser.add_argument("--max-psnr-drop", type=float, default=0.02)
    parser.add_argument("--max-ssim-drop", type=float, default=0.0003)
    parser.add_argument("--max-lpips-increase", type=float, default=0.001)
    parser.add_argument("--max-latency-ms", type=float, default=15.0)
    return parser.parse_args()


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def latency(document: dict) -> float:
    return float(document["milliseconds_per_image"]["p50"])


def main() -> None:
    args = parse_args()
    control = read(args.control_metrics)
    candidate = read(args.candidate_metrics)
    control_stress, candidate_stress = read(args.control_stress), read(args.candidate_stress)
    fixed_ood = read(args.ood_fixed_policy)
    randomized_ood = read(args.ood_randomized_policy)
    control_defects, candidate_defects = read(args.control_defects), read(args.candidate_defects)
    control_benchmark, candidate_benchmark = read(args.control_benchmark), read(args.candidate_benchmark)
    uncertainty = read(args.uncertainty_validation)

    checks = {
        "official_psnr_guard": candidate["psnr"] >= control["psnr"] - args.max_psnr_drop,
        "official_ssim_guard": candidate["ssim"] >= control["ssim"] - args.max_ssim_drop,
        "official_lpips_guard": candidate["lpips"] <= control["lpips"] + args.max_lpips_increase,
        "stress_psnr_improves": candidate_stress["macro_average"]["psnr"] > control_stress["macro_average"]["psnr"],
        "stress_ssim_improves": candidate_stress["macro_average"]["ssim"] > control_stress["macro_average"]["ssim"],
        "both_orders_psnr_improve": all(
            candidate_stress["scenarios"][name]["psnr"] > control_stress["scenarios"][name]["psnr"]
            for name in ("noise_before_downsample", "downsample_before_noise")
        ),
        "randomized_policy_ood_psnr_improves": randomized_ood["psnr"] > fixed_ood["psnr"],
        "randomized_policy_ood_ssim_improves": randomized_ood["ssim"] > fixed_ood["ssim"],
        "defect_f1_improves": candidate_defects["summary"]["f1"]["mean"] > control_defects["summary"]["f1"]["mean"],
        "false_patterns_not_worse": candidate_defects["summary"]["false_pattern_rate"]["mean"] <= control_defects["summary"]["false_pattern_rate"]["mean"] + 1e-6,
        "latency_guard": latency(candidate_benchmark) <= args.max_latency_ms,
        "uncertainty_ranks_error": (
            uncertainty["uncertainty_error_spearman"] > 0.20
            and uncertainty["highest_uncertainty_mae"]
            > uncertainty["lowest_uncertainty_mae"]
        ),
    }
    passed = all(checks.values())
    result = {
        "passed": passed,
        "checks": checks,
        "thresholds": {
            "max_psnr_drop": args.max_psnr_drop,
            "max_ssim_drop": args.max_ssim_drop,
            "max_lpips_increase": args.max_lpips_increase,
            "max_latency_ms": args.max_latency_ms,
        },
        "candidate_weights": str(args.candidate_weights),
        "ood_evidence_scope": (
            "Independent matched cluster-split policy ablation; the promoted "
            "checkpoint itself is not assigned an OOD score."
        ),
        "promotion_target": str(args.promote_to) if args.promote_to else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if passed and args.promote_to is not None:
        args.promote_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate_weights, args.promote_to)
        print(f"PROMOTED {args.candidate_weights} -> {args.promote_to}")
    elif not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

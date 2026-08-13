#!/usr/bin/env python3
"""Fail-fast audit for the public KLA submission repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


REQUIRED_FILES = (
    "README.md",
    "Dockerfile",
    "requirements.txt",
    "requirements.runtime.txt",
    "inference.py",
    "train.py",
    "evaluate.py",
    "validate_submission.py",
    "evaluate_defects.py",
    "inference_uncertainty.py",
    "make_ood_split.py",
    "benchmark_resolutions.py",
    "promote_robust_candidate.py",
    "make_robustness_figures.py",
    "make_output_manifest.py",
    "REFERENCES.md",
    "LICENSE",
    "weights/final.pt",
    "outputs/output_manifest.json",
)
SECRET_PATTERNS = {
    "private IP": re.compile(r"\b10\.30\.161\.73\b"),
    "GitHub token": re.compile(r"\bgh[opsu]_[A-Za-z0-9_]{20,}\b"),
    "password assignment": re.compile(r"(?i)password\s*[:=]\s*\S+"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--expected-outputs", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise SystemExit("Missing required files: " + ", ".join(missing))

    freeze = (root / "requirements.txt").read_text().splitlines()
    if len(freeze) < 100:
        raise SystemExit("requirements.txt does not look like a complete environment freeze")
    forbidden_freeze_text = ("== PyTorch ==", "NVIDIA Release", "GOVERNING TERMS", "CUDA failed")
    if any(marker in line for line in freeze for marker in forbidden_freeze_text):
        raise SystemExit("requirements.txt contains container startup output")

    manifest = json.loads((root / "outputs/output_manifest.json").read_text())
    if manifest.get("count") != args.expected_outputs:
        raise SystemExit(f"Manifest count is {manifest.get('count')}, expected {args.expected_outputs}")
    outputs = sorted((root / "outputs/restored").glob("*.npy"))
    if len(outputs) != args.expected_outputs:
        raise SystemExit(f"Found {len(outputs)} restored files, expected {args.expected_outputs}")
    aggregate = hashlib.sha256()
    for entry, path in zip(manifest["files"], outputs, strict=True):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if entry["filename"] != path.name or entry["sha256"] != digest:
            raise SystemExit(f"Manifest mismatch for {path.name}")
        aggregate.update(f"{path.name} {digest}\n".encode())
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise SystemExit("Aggregate output hash mismatch")

    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout.split(b"\0")
    findings = []
    for encoded in tracked:
        if not encoded:
            continue
        relative = encoded.decode(errors="replace")
        path = root / relative
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {relative}")
    if findings:
        raise SystemExit("Potential secrets/private data:\n" + "\n".join(findings))

    check_ignore = subprocess.run(
        ["git", "check-ignore", "-q", "outputs/restored/000000.npy"], cwd=root
    )
    if check_ignore.returncode == 0:
        raise SystemExit("Restored outputs are still ignored by Git")
    print(
        f"OK: required_files={len(REQUIRED_FILES)} outputs={len(outputs)} "
        f"aggregate_sha256={manifest['aggregate_sha256']} secret_scan=passed"
    )


if __name__ == "__main__":
    main()

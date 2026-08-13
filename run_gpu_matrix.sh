#!/usr/bin/env bash
# Bounded KLA experiment matrix. Run from ~/Semicon_KLA on the NVIDIA host.
set -euo pipefail

IMAGE="${KLA_IMAGE:-kla-restorenet:latest}"
ROOT="$(pwd)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building $IMAGE from the pinned NVIDIA NGC base image..."
  docker build -t "$IMAGE" .
fi

run_container() {
  docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e PYTHONUNBUFFERED=1 \
    -v "$ROOT":/workspace/project \
    -w /workspace/project \
    "$IMAGE" "$@"
}

mkdir -p weights/sweep outputs/experiments logs

case "${1:-}" in
  tta)
    run_container bash -lc '
      python evaluate.py --method model --split val \
        --input-dir data/train/NoisyLR --gt-dir data/train/GT \
        --weights weights/final.pt --self-ensemble x4 --batch-size 8 \
        --device cuda --lpips \
        --json-output outputs/experiments/v2_x4.json &&
      python evaluate.py --method model --split val \
        --input-dir data/train/NoisyLR --gt-dir data/train/GT \
        --weights weights/final.pt --self-ensemble x8 --batch-size 8 \
        --device cuda --lpips \
        --json-output outputs/experiments/v2_x8.json &&
      python benchmark.py --weights weights/final.pt --self-ensemble x4 \
        --batch-size 1 --device cuda \
        --json-output outputs/experiments/v2_x4_benchmark.json &&
      python benchmark.py --weights weights/final.pt --self-ensemble x8 \
        --batch-size 1 --device cuda \
        --json-output outputs/experiments/v2_x8_benchmark.json
    '
    ;;
  seed3407|seed8119)
    SEED="${1#seed}"
    run_container bash -lc "python -u train.py \
      --epochs 30 --batch-size 8 --workers 4 --width 48 --blocks 12 \
      --learning-rate 1e-4 --synthetic-probability 0.2 \
      --synthetic-policy fixed --ema-decay 0.999 --gradient-clip 1.0 \
      --seed $SEED --device cuda --output-dir weights/sweep/seed$SEED \
      2>&1 | tee logs/train_seed$SEED.log"
    ;;
  width64)
    run_container bash -lc 'python -u train.py \
      --epochs 30 --batch-size 8 --workers 4 --width 64 --blocks 12 \
      --learning-rate 1e-4 --synthetic-probability 0.2 \
      --synthetic-policy fixed --ema-decay 0.999 --gradient-clip 1.0 \
      --seed 2026 --device cuda --output-dir weights/sweep/width64 \
      2>&1 | tee logs/train_width64.log'
    ;;
  pixelheavy)
    run_container bash -lc 'python -u train.py \
      --epochs 30 --batch-size 8 --workers 4 --width 48 --blocks 12 \
      --learning-rate 1e-4 --synthetic-probability 0.2 \
      --synthetic-policy fixed --pixel-weight 0.8 --ssim-weight 0.15 \
      --edge-weight 0.05 --ema-decay 0.999 --gradient-clip 1.0 \
      --seed 2026 --device cuda --output-dir weights/sweep/pixelheavy \
      2>&1 | tee logs/train_pixelheavy.log'
    ;;
  v4a-pilot)
    run_container bash -lc 'python -u train.py \
      --variant v4a --initialize-from weights/final.pt \
      --epochs 5 --batch-size 8 --workers 4 --width 48 --blocks 12 \
      --learning-rate 5e-5 --synthetic-probability 0.2 \
      --synthetic-policy fixed --ema-decay 0.995 --gradient-clip 1.0 \
      --seed 2026 --device cuda --output-dir weights/v4a_pilot \
      2>&1 | tee logs/train_v4a_pilot.log'
    ;;
  v4a-evaluate)
    run_container python evaluate.py --method model --split val \
      --input-dir data/train/NoisyLR --gt-dir data/train/GT \
      --weights weights/v4a_pilot/best_balanced.pt \
      --batch-size 8 --device cuda --lpips \
      --json-output outputs/experiments/v4a_pilot.json \
      --per-image-output outputs/experiments/v4a_pilot_per_image.csv
    run_container python evaluate_stress.py \
      --weights weights/v4a_pilot/best_balanced.pt --device cuda \
      --json-output outputs/experiments/v4a_pilot_stress.json
    run_container python benchmark.py \
      --weights weights/v4a_pilot/best_balanced.pt --batch-size 1 \
      --device cuda \
      --json-output outputs/experiments/v4a_pilot_benchmark.json
    ;;
  split3407)
    run_container bash -lc 'python -u train.py \
      --epochs 30 --batch-size 8 --workers 4 --width 48 --blocks 12 \
      --learning-rate 1e-4 --synthetic-probability 0.2 \
      --synthetic-policy fixed --split-seed 3407 --ema-decay 0.999 \
      --gradient-clip 1.0 --seed 2026 --device cuda \
      --output-dir weights/sweep/split3407 \
      2>&1 | tee logs/train_split3407.log'
    ;;
  evaluate)
    for candidate in seed3407 seed8119 width64 pixelheavy; do
      run_container python evaluate.py --method model --split val \
        --input-dir data/train/NoisyLR --gt-dir data/train/GT \
        --weights "weights/sweep/$candidate/best_balanced.pt" \
        --batch-size 8 --device cuda --lpips \
        --json-output "outputs/experiments/${candidate}.json" \
        --per-image-output "outputs/experiments/${candidate}_per_image.csv"
      run_container python evaluate_stress.py \
        --weights "weights/sweep/$candidate/best_balanced.pt" --device cuda \
        --json-output "outputs/experiments/${candidate}_stress.json"
      run_container python benchmark.py \
        --weights "weights/sweep/$candidate/best_balanced.pt" --batch-size 1 \
        --device cuda \
        --json-output "outputs/experiments/${candidate}_benchmark.json"
    done
    run_container python select_candidate.py
    ;;
  all-data)
    run_container bash -lc 'python -u train.py \
      --train-all --epochs 24 --batch-size 8 --workers 4 \
      --width 48 --blocks 12 --learning-rate 1e-4 \
      --synthetic-probability 0.2 --synthetic-policy fixed \
      --ema-decay 0.999 --gradient-clip 1.0 --seed 2026 --device cuda \
      --output-dir weights/all_data \
      2>&1 | tee logs/train_all_data.log'
    ;;
  *)
    echo "Usage: $0 {tta|seed3407|seed8119|width64|pixelheavy|v4a-pilot|v4a-evaluate|split3407|evaluate|all-data}" >&2
    exit 2
    ;;
esac

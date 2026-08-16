#!/usr/bin/env bash
# Bounded KLA experiment matrix. Run from ~/Semicon_KLA on the NVIDIA host.
set -euo pipefail

IMAGE="${OUR_MODEL_IMAGE:-our-model:latest}"
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

RUN_LOCK_DIR=""

release_run_lock() {
  if [[ -n "${RUN_LOCK_DIR:-}" ]]; then
    rmdir "$RUN_LOCK_DIR" 2>/dev/null || true
    RUN_LOCK_DIR=""
  fi
}

acquire_run_lock() {
  RUN_LOCK_DIR="$1"
  if ! mkdir "$RUN_LOCK_DIR" 2>/dev/null; then
    echo "Another run owns lock $RUN_LOCK_DIR; refusing duplicate launch." >&2
    RUN_LOCK_DIR=""
    exit 3
  fi
  trap release_run_lock EXIT INT TERM
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
  v4a-ablation)
    run_container bash -lc '
      python -u train.py --variant v2 --initialize-from weights/final.pt \
        --epochs 5 --batch-size 8 --workers 4 --width 48 --blocks 12 \
        --learning-rate 5e-5 --synthetic-probability 0.2 \
        --synthetic-policy fixed --ema-decay 0.995 --gradient-clip 1.0 \
        --seed 2026 --device cuda --output-dir weights/v4a_ablation/v2_control \
        2>&1 | tee logs/train_v4a_ablation_v2_control.log &&
      python -u train.py --variant v4a --initialize-from weights/final.pt \
        --epochs 5 --batch-size 8 --workers 4 --width 48 --blocks 12 \
        --learning-rate 5e-5 --synthetic-probability 0.2 \
        --synthetic-policy fixed --ema-decay 0.995 --gradient-clip 1.0 \
        --seed 2026 --device cuda --output-dir weights/v4a_ablation/v4a \
        2>&1 | tee logs/train_v4a_ablation_v4a.log
    '
    for candidate in frozen_v2 v2_control v4a; do
      case "$candidate" in
        frozen_v2) checkpoint="weights/final.pt" ;;
        v2_control) checkpoint="weights/v4a_ablation/v2_control/best_balanced.pt" ;;
        v4a) checkpoint="weights/v4a_ablation/v4a/best_balanced.pt" ;;
      esac
      run_container python evaluate.py --method model --split val \
        --input-dir data/train/NoisyLR --gt-dir data/train/GT \
        --weights "$checkpoint" --batch-size 8 --device cuda --lpips \
        --json-output "outputs/experiments/ablation_${candidate}.json" \
        --per-image-output "outputs/experiments/ablation_${candidate}_per_image.csv"
      run_container python evaluate_stress.py --weights "$checkpoint" \
        --device cuda \
        --json-output "outputs/experiments/ablation_${candidate}_stress.json"
      run_container python benchmark.py --weights "$checkpoint" --batch-size 1 \
        --device cuda \
        --json-output "outputs/experiments/ablation_${candidate}_benchmark.json"
    done
    run_container python compare_paired.py \
      --baseline outputs/experiments/ablation_frozen_v2_per_image.csv \
      --control outputs/experiments/ablation_v2_control_per_image.csv \
      --candidate outputs/experiments/ablation_v4a_per_image.csv \
      --candidate-name v4a \
      --output results/v4a_paired_ablation.json
    ;;
  v4b-ablation)
    run_container bash -lc '
      python -u train.py --variant v2 --initialize-from weights/final.pt \
        --epochs 5 --batch-size 8 --workers 4 --width 48 --blocks 12 \
        --learning-rate 3e-5 --synthetic-probability 0.2 \
        --synthetic-policy randomized --ema-decay 0.995 --gradient-clip 1.0 \
        --seed 2026 --device cuda --output-dir weights/v4b_ablation/v2_control \
        2>&1 | tee logs/train_v4b_ablation_v2_control.log &&
      python -u train.py --variant v4b --initialize-from weights/final.pt \
        --frequency-width 24 --frequency-blocks 2 \
        --epochs 5 --batch-size 8 --workers 4 --width 48 --blocks 12 \
        --learning-rate 3e-5 --synthetic-probability 0.2 \
        --synthetic-policy randomized --ema-decay 0.995 --gradient-clip 1.0 \
        --seed 2026 --device cuda --output-dir weights/v4b_ablation/v4b \
        2>&1 | tee logs/train_v4b_ablation_v4b.log
    '
    for candidate in frozen_v2 v2_control v4b; do
      case "$candidate" in
        frozen_v2) checkpoint="weights/final.pt" ;;
        v2_control) checkpoint="weights/v4b_ablation/v2_control/best_balanced.pt" ;;
        v4b) checkpoint="weights/v4b_ablation/v4b/best_balanced.pt" ;;
      esac
      run_container python evaluate.py --method model --split val \
        --input-dir data/train/NoisyLR --gt-dir data/train/GT \
        --weights "$checkpoint" --batch-size 8 --device cuda --lpips \
        --json-output "outputs/experiments/v4b_${candidate}.json" \
        --per-image-output "outputs/experiments/v4b_${candidate}_per_image.csv"
      run_container python evaluate_stress.py --weights "$checkpoint" \
        --device cuda \
        --json-output "outputs/experiments/v4b_${candidate}_stress.json"
      run_container python benchmark.py --weights "$checkpoint" --batch-size 1 \
        --device cuda \
        --json-output "outputs/experiments/v4b_${candidate}_benchmark.json"
    done
    run_container python compare_paired.py \
      --baseline outputs/experiments/v4b_frozen_v2_per_image.csv \
      --control outputs/experiments/v4b_v2_control_per_image.csv \
      --candidate outputs/experiments/v4b_v4b_per_image.csv \
      --candidate-name v4b \
      --output results/v4b_paired_ablation.json
    ;;
  v4b-staged)
    echo "v4b-staged is deprecated after an unstable pilot; use v4b-v2." >&2
    exit 2
    ;;
  v4b-v2)
    acquire_run_lock "$ROOT/.v4b_v2.lock"
    run_container bash -lc 'set -o pipefail
      python -u train.py --variant v4b --initialize-from weights/final.pt \
        --frequency-width 24 --frequency-blocks 2 \
        --epochs 12 --freeze-backbone-epochs 2 \
        --backbone-learning-rate 2e-6 --branch-learning-rate 5e-6 \
        --preservation-weights weights/final.pt --preservation-weight 0.10 \
        --collapse-guard-psnr-drop 0.10 \
        --early-stopping-patience 4 --early-stopping-min-delta 0.001 \
        --batch-size 8 --workers 4 --width 48 --blocks 12 \
        --synthetic-probability 0.2 --synthetic-policy randomized \
        --ema-decay 0.995 --gradient-clip 1.0 --seed 2026 --device cuda \
        --output-dir weights/v4b_v2 \
        2>&1 | tee logs/train_v4b_v2.log
    '
    for candidate in final_v2 staged_v4b; do
      case "$candidate" in
        final_v2) checkpoint="weights/final.pt" ;;
        staged_v4b) checkpoint="weights/v4b_v2/best_balanced.pt" ;;
      esac
      run_container python evaluate.py --method model --split val \
        --input-dir data/train/NoisyLR --gt-dir data/train/GT \
        --weights "$checkpoint" --batch-size 8 --device cuda --lpips \
        --json-output "outputs/experiments/v4b_v2_${candidate}.json" \
        --per-image-output "outputs/experiments/v4b_v2_${candidate}_per_image.csv"
      run_container python evaluate_stress.py --weights "$checkpoint" \
        --device cuda \
        --json-output "outputs/experiments/v4b_v2_${candidate}_stress.json"
      run_container python benchmark.py --weights "$checkpoint" --batch-size 1 \
        --device cuda \
        --json-output "outputs/experiments/v4b_v2_${candidate}_benchmark.json"
    done
    run_container python compare_paired.py \
      --baseline outputs/experiments/v4b_v2_final_v2_per_image.csv \
      --control outputs/experiments/v4b_v2_final_v2_per_image.csv \
      --candidate outputs/experiments/v4b_v2_staged_v4b_per_image.csv \
      --candidate-name v4b_v2 \
      --output results/v4b_v2_vs_final.json
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
  robustness-ood)
    acquire_run_lock "$ROOT/.robustness_ood.lock"
    run_container python make_ood_split.py \
      --clusters 10 --target-size 320 --seed 260813 \
      --output splits/ood_cluster.json
    run_container bash -lc 'set -o pipefail
      python -u train.py --split-manifest splits/ood_cluster.json \
        --epochs 30 --batch-size 8 --workers 4 --width 48 --blocks 12 \
        --learning-rate 1e-4 --synthetic-probability 0.05 \
        --synthetic-probability-final 0.20 --synthetic-policy fixed \
        --ema-decay 0.999 --gradient-clip 1.0 --seed 2026 --device cuda \
        --output-dir weights/robustness_ood/fixed_control \
        2>&1 | tee logs/train_robustness_ood_fixed.log &&
      python -u train.py --split-manifest splits/ood_cluster.json \
        --epochs 30 --batch-size 8 --workers 4 --width 48 --blocks 12 \
        --learning-rate 1e-4 --synthetic-probability 0.05 \
        --synthetic-probability-final 0.20 --synthetic-policy randomized \
        --ema-decay 0.999 --gradient-clip 1.0 --seed 2026 --device cuda \
        --output-dir weights/robustness_ood/randomized \
        2>&1 | tee logs/train_robustness_ood_randomized.log
    '
    ;;
  robustness-candidate)
    acquire_run_lock "$ROOT/.robustness_candidate.lock"
    run_container bash -lc 'set -o pipefail
      python -u train.py --variant v2 --initialize-from weights/final.pt \
        --epochs 8 --batch-size 8 --workers 4 --width 48 --blocks 12 \
        --learning-rate 5e-6 --synthetic-probability 0.05 \
        --synthetic-probability-final 0.15 --synthetic-policy randomized \
        --preservation-weights weights/final.pt --preservation-weight 0.10 \
        --collapse-guard-psnr-drop 0.05 --early-stopping-patience 3 \
        --early-stopping-min-delta 0.001 --ema-decay 0.995 \
        --gradient-clip 1.0 --seed 2026 --device cuda \
        --output-dir weights/robustness_candidate \
        2>&1 | tee logs/train_robustness_candidate.log
    '
    ;;
  robustness-evaluate)
    acquire_run_lock "$ROOT/.robustness_evaluate.lock"
    mkdir -p outputs/robustness results weights/promoted
    for candidate in final challenger; do
      case "$candidate" in
        final) checkpoint="weights/final.pt" ;;
        challenger) checkpoint="weights/robustness_candidate/best_balanced.pt" ;;
      esac
      run_container python evaluate.py --method model --split val \
        --input-dir data/train/NoisyLR --gt-dir data/train/GT \
        --weights "$checkpoint" --batch-size 8 --device cuda --lpips \
        --json-output "outputs/robustness/${candidate}_official.json" \
        --per-image-output "outputs/robustness/${candidate}_official.csv"
      run_container python evaluate_stress.py --weights "$checkpoint" \
        --device cuda --json-output "outputs/robustness/${candidate}_stress.json"
      run_container python evaluate_defects.py --weights "$checkpoint" \
        --device cuda --limit 80 \
        --json-output "outputs/robustness/${candidate}_defects.json"
      run_container python benchmark.py --weights "$checkpoint" --batch-size 1 \
        --device cuda --json-output "outputs/robustness/${candidate}_benchmark.json"
    done
    run_container python evaluate_defects.py --method bicubic --device cuda --limit 80 \
      --json-output outputs/robustness/bicubic_defects.json
    for policy in fixed randomized; do
      case "$policy" in
        fixed) checkpoint="weights/robustness_ood/fixed_control/best_balanced.pt" ;;
        randomized) checkpoint="weights/robustness_ood/randomized/best_balanced.pt" ;;
      esac
      run_container python evaluate.py --method model --split val \
        --names-manifest splits/ood_cluster.json --manifest-key val_names \
        --input-dir data/train/NoisyLR --gt-dir data/train/GT \
        --weights "$checkpoint" --batch-size 8 --device cuda --lpips \
        --json-output "outputs/robustness/ood_${policy}.json" \
        --per-image-output "outputs/robustness/ood_${policy}.csv"
    done
    run_container python benchmark_resolutions.py --weights weights/final.pt \
      --device cuda --json-output outputs/robustness/final_resolutions.json
    run_container python evaluate_uncertainty.py --weights weights/final.pt \
      --self-ensemble x8 --batch-size 4 --device cuda \
      --json-output outputs/robustness/uncertainty_validation.json
    run_container python promote_robust_candidate.py \
      --control-metrics outputs/robustness/final_official.json \
      --candidate-metrics outputs/robustness/challenger_official.json \
      --control-stress outputs/robustness/final_stress.json \
      --candidate-stress outputs/robustness/challenger_stress.json \
      --ood-fixed-policy outputs/robustness/ood_fixed.json \
      --ood-randomized-policy outputs/robustness/ood_randomized.json \
      --control-defects outputs/robustness/final_defects.json \
      --candidate-defects outputs/robustness/challenger_defects.json \
      --control-benchmark outputs/robustness/final_benchmark.json \
      --candidate-benchmark outputs/robustness/challenger_benchmark.json \
      --uncertainty-validation outputs/robustness/uncertainty_validation.json \
      --candidate-weights weights/robustness_candidate/best_balanced.pt \
      --report results/robustness_promotion.json \
      --promote-to weights/promoted/robustness_candidate.pt
    ;;
  robustness-uncertainty)
    acquire_run_lock "$ROOT/.robustness_uncertainty.lock"
    run_container python inference_uncertainty.py \
      --input-dir data/test/NoisyLR \
      --output-dir outputs/robustness/restored_x8 \
      --uncertainty-dir outputs/robustness/uncertainty \
      --weights weights/final.pt --self-ensemble x8 --batch-size 4 \
      --device cuda --summary-output outputs/robustness/uncertainty_summary.json
    ;;
  robustness-figures)
    run_container python make_robustness_figures.py \
      --results-dir outputs/robustness --input-dir data/test/NoisyLR \
      --output-dir figures/robustness
    ;;
  presentation-figures)
    run_container python make_figures.py \
      --weights weights/final.pt \
      --history weights/v2/history.json \
      --lr-dir data/train/NoisyLR --gt-dir data/train/GT \
      --batch-size 8 --workers 4 --device cuda \
      --output-dir figures
    ;;
  robustness-all)
    "$0" robustness-ood
    "$0" robustness-candidate
    "$0" robustness-evaluate
    "$0" robustness-uncertainty
    "$0" robustness-figures
    ;;
  *)
    echo "Usage: $0 {tta|seed3407|seed8119|width64|pixelheavy|v4a-pilot|v4a-evaluate|v4a-ablation|v4b-ablation|v4b-v2|split3407|evaluate|all-data|robustness-ood|robustness-candidate|robustness-evaluate|robustness-uncertainty|robustness-figures|robustness-all|presentation-figures}" >&2
    exit 2
    ;;
esac

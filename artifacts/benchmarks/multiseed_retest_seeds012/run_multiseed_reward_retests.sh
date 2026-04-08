#!/usr/bin/env bash
set -euo pipefail

ROOT=/tmp/milabench/artifacts/benchmarks/multiseed_retest_seeds012
PYTHON=/tmp/results/venv/torch/bin/python
MAIN=/tmp/milabench/benchmarks/purejaxrl/main.py
SEEDS=(0 1 2)
EXPERIMENTS=(control target_fold bf16 fp16 trainint20)

mkdir -p "$ROOT"

run_case() {
  local experiment="$1"
  local seed="$2"
  shift 2

  local out_dir="$ROOT/$experiment/seed${seed}"
  mkdir -p "$out_dir"

  echo "[multiseed-retest] $experiment seed=$seed"
  "$@" > "$out_dir/stdout.txt" 2> "$out_dir/stderr.txt"
}

base_args() {
  local seed="$1"
  printf '%s\n' \
    dqn \
    --seed "$seed" \
    --num_envs 128 \
    --buffer_size 131072 \
    --buffer_batch_size 65536 \
    --env_name SpaceInvaders-MinAtar \
    --training_interval 10 \
    --total_timesteps 2000000
}

cd /tmp/milabench
export MPLCONFIGDIR=/tmp/mpl

for seed in "${SEEDS[@]}"; do
  readarray -t args < <(base_args "$seed")

  run_case control "$seed" \
    env PUREJAXRL_DQN_FOLD_TARGET_UPDATE=0 \
    "$PYTHON" "$MAIN" "${args[@]}"

  run_case target_fold "$seed" \
    env PUREJAXRL_DQN_FOLD_TARGET_UPDATE=1 \
    "$PYTHON" "$MAIN" "${args[@]}"

  run_case bf16 "$seed" \
    env PUREJAXRL_DQN_FOLD_TARGET_UPDATE=0 \
    "$PYTHON" "$MAIN" "${args[@]}" --dtype bf16

  run_case fp16 "$seed" \
    env PUREJAXRL_DQN_FOLD_TARGET_UPDATE=0 \
    "$PYTHON" "$MAIN" "${args[@]}" --dtype fp16

  run_case trainint20 "$seed" \
    env PUREJAXRL_DQN_FOLD_TARGET_UPDATE=0 \
    "$PYTHON" "$MAIN" dqn \
    --seed "$seed" \
    --num_envs 128 \
    --buffer_size 131072 \
    --buffer_batch_size 65536 \
    --env_name SpaceInvaders-MinAtar \
    --training_interval 20 \
    --total_timesteps 2000000
done

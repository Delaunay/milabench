#!/usr/bin/env bash
set -euo pipefail

ROOT=/tmp/milabench/artifacts/benchmarks/seeded_retest_seed0
PYTHON=/tmp/results/venv/torch/bin/python
MAIN=/tmp/milabench/benchmarks/purejaxrl/main.py

mkdir -p \
  "$ROOT/control" \
  "$ROOT/target_fold" \
  "$ROOT/bf16" \
  "$ROOT/fp16" \
  "$ROOT/numenvs192" \
  "$ROOT/numenvs256" \
  "$ROOT/trainint20"

run_case() {
  local name="$1"
  shift
  echo "[seeded-retest] $name"
  "$@" > "$ROOT/$name/stdout.txt" 2> "$ROOT/$name/stderr.txt"
}

cd /tmp/milabench
export MPLCONFIGDIR=/tmp/mpl

run_case control \
  env PUREJAXRL_DQN_FOLD_TARGET_UPDATE=0 \
  "$PYTHON" "$MAIN" dqn \
  --seed 0 \
  --num_envs 128 \
  --buffer_size 131072 \
  --buffer_batch_size 65536 \
  --env_name SpaceInvaders-MinAtar \
  --training_interval 10 \
  --total_timesteps 2000000

run_case target_fold \
  env PUREJAXRL_DQN_FOLD_TARGET_UPDATE=1 \
  "$PYTHON" "$MAIN" dqn \
  --seed 0 \
  --num_envs 128 \
  --buffer_size 131072 \
  --buffer_batch_size 65536 \
  --env_name SpaceInvaders-MinAtar \
  --training_interval 10 \
  --total_timesteps 2000000

run_case bf16 \
  env PUREJAXRL_DQN_FOLD_TARGET_UPDATE=0 \
  "$PYTHON" "$MAIN" dqn \
  --seed 0 \
  --num_envs 128 \
  --buffer_size 131072 \
  --buffer_batch_size 65536 \
  --env_name SpaceInvaders-MinAtar \
  --training_interval 10 \
  --total_timesteps 2000000 \
  --dtype bf16

run_case fp16 \
  env PUREJAXRL_DQN_FOLD_TARGET_UPDATE=0 \
  "$PYTHON" "$MAIN" dqn \
  --seed 0 \
  --num_envs 128 \
  --buffer_size 131072 \
  --buffer_batch_size 65536 \
  --env_name SpaceInvaders-MinAtar \
  --training_interval 10 \
  --total_timesteps 2000000 \
  --dtype fp16

run_case numenvs192 \
  env PUREJAXRL_DQN_FOLD_TARGET_UPDATE=0 \
  "$PYTHON" "$MAIN" dqn \
  --seed 0 \
  --num_envs 192 \
  --buffer_size 131072 \
  --buffer_batch_size 65536 \
  --env_name SpaceInvaders-MinAtar \
  --training_interval 10 \
  --total_timesteps 2000000

run_case numenvs256 \
  env PUREJAXRL_DQN_FOLD_TARGET_UPDATE=0 \
  "$PYTHON" "$MAIN" dqn \
  --seed 0 \
  --num_envs 256 \
  --buffer_size 131072 \
  --buffer_batch_size 65536 \
  --env_name SpaceInvaders-MinAtar \
  --training_interval 10 \
  --total_timesteps 2000000

run_case trainint20 \
  env PUREJAXRL_DQN_FOLD_TARGET_UPDATE=0 \
  "$PYTHON" "$MAIN" dqn \
  --seed 0 \
  --num_envs 128 \
  --buffer_size 131072 \
  --buffer_batch_size 65536 \
  --env_name SpaceInvaders-MinAtar \
  --training_interval 20 \
  --total_timesteps 2000000

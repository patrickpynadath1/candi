#!/usr/bin/env bash
set -euo pipefail

gpu_id=${1:-5}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_dir="${repo_root}/new_sudoku_runs/sfm_truncated_adaptive"
mkdir -p "${run_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config \
  +experiment=sudoku_sfm_truncated_adaptive \
  wandb.project=sudoku-sfm-paper-retrain \
  wandb.name=sfm-truncated-adaptive-easy-20k \
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=${run_dir}" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  "hydra.run.dir=${run_dir}"

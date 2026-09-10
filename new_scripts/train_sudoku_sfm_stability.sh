#!/usr/bin/env bash
set -euo pipefail

gpu_id=${1:-0}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=${2:-"${repo_root}/outputs/sfm_stability/sudoku_hard/paper_truncated_adaptive/seed_1"}

mkdir -p "${run_dir}"

exec env \
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  OMP_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config \
  +experiment=sudoku_sfm_paper_hard \
  data.cache_dir=/home/patrick/.cache/duo/data_cache \
  trainer.devices=1 \
  trainer.num_nodes=1 \
  wandb.project=reasoning-embedding-pure-ablations \
  wandb.group=sfm-stability \
  wandb.job_type=train \
  wandb.name=sfm-stability-sudoku-hard-paper-truncated-adaptive-seed-1 \
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=${run_dir}" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  "hydra.run.dir=${run_dir}"

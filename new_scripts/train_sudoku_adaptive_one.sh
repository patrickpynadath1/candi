#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_ID ALGORITHM" >&2
  exit 2
fi
gpu_id=$1
algorithm=$2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
label=${algorithm}
experiment="sudoku_${algorithm}_adaptive"
if [[ "${algorithm}" == "ar" ]]; then
  label=ar_control
  experiment=sudoku_ar_adaptive_control
fi
run_dir="${repo_root}/new_sudoku_runs/adaptive/${label}"
mkdir -p "${run_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config \
  "+experiment=${experiment}" \
  wandb.project=sudoku-adaptive-schedule-20k \
  "wandb.name=${label}-sudoku-adaptive-20k" \
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=${run_dir}" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  "hydra.run.dir=${run_dir}"

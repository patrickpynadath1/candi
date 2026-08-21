#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_ID {mdlm|candi}" >&2
  exit 2
fi

gpu_id=$1
algorithm=$2
if [[ "${algorithm}" != "mdlm" && "${algorithm}" != "candi" ]]; then
  echo "algorithm must be mdlm or candi" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
run_dir="${storage_root}/new_sudoku_runs/hard_eb_0p1/${algorithm}"
dataset_cache="${storage_root}/data_cache"
mkdir -p "${run_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config \
  "+experiment=sudoku_${algorithm}_hard_eb" \
  "data.cache_dir=${dataset_cache}" \
  trainer.max_steps=20000 \
  trainer.val_check_interval=20000 \
  wandb.project=sudoku-hard-tiny-dit-20k \
  wandb.group=hard-eb-0p1 \
  "wandb.name=${algorithm}-tiny-dit-sudoku-hard-eb-0p1-20k" \
  "wandb.tags=[sudoku,hard,${algorithm},tiny-dit,entropy-bounded,bound-0.1,20k-steps]" \
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=${run_dir}" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  +trainer.check_val_every_n_epoch=null \
  "hydra.run.dir=${run_dir}"

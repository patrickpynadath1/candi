#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU_ID RUN_NAME EXPERIMENT [OVERRIDES ...]" >&2
  exit 2
fi

gpu_id=$1
run_name=$2
experiment=$3
shift 3
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="${repo_root}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="/home/patrick/duo/.venv/bin/python"
fi
run_dir="${repo_root}/new_candi_curve_runs/${run_name}"
mkdir -p "${run_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" "${repo_root}/main.py" \
  --config-name=config \
  "+experiment=${experiment}" \
  wandb.project=sudoku-candi-curves-5k \
  "wandb.name=${run_name}" \
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=${run_dir}" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  data.cache_dir=/home/patrick/duo/data_cache \
  "hydra.run.dir=${run_dir}" \
    "$@"

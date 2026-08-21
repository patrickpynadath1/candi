#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID {sflm|uniform} [HYDRA_OVERRIDE ...]" >&2
  exit 2
fi

gpu_id=$1
method=$2
shift 2

case "${method}" in
  sflm) experiment=sudoku_sfm_paper_hard ;;
  uniform) experiment=sudoku_uniform_hard_multi_sampler ;;
  *) echo "method must be sflm or uniform" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
run_mode=${DUO_RUN_MODE:-full}
run_dir="${storage_root}/new_sudoku_runs/sflm_uniform_paper_hard_${run_mode}/${method}"
dataset_cache="${storage_root}/data_cache"
mkdir -p "${run_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config \
  "+experiment=${experiment}" \
  "data.cache_dir=${dataset_cache}" \
  wandb.project=sudoku-blurring-hard-20k \
  wandb.group=sflm-uniform-paper-hard \
  "wandb.name=${method}-sudoku-hard-paper-20k" \
  "wandb.tags=[sudoku,hard,${method},paper,20k-steps]" \
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=${run_dir}" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  "hydra.run.dir=${run_dir}" \
  "$@"

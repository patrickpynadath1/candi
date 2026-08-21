#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 ORIGINAL_GPU TRANSPOSED_GPU [ve|flow_matching]" >&2
  exit 2
fi

original_gpu=$1
transposed_gpu=$2
continuous_path=${3:-flow_matching}
if [[ "${continuous_path}" != "ve" && "${continuous_path}" != "flow_matching" ]]; then
  echo "continuous path must be 've' or 'flow_matching'" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
run_root="${storage_root}/new_sudoku_runs/embedding_diffusion_${continuous_path}"
dataset_cache="${storage_root}/data_cache"
mkdir -p "${run_root}"

run_experiment() {
  local gpu_id=$1
  local experiment=$2
  local label=$3
  local run_dir="${run_root}/${label}"
  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
    --config-name=config \
    "+experiment=${experiment}" \
    "data.cache_dir=${dataset_cache}" \
    "algo.continuous_path=${continuous_path}" \
    wandb.project=sudoku-hard-tiny-dit-20k \
    wandb.group=embedding-diffusion-axis-comparison \
    "wandb.name=${label}-embedding-diffusion-${continuous_path}-sudoku-hard-20k" \
    "wandb.tags=[sudoku,hard,embedding-diffusion,${label},${continuous_path},tiny-dit,20k-steps]" \
    checkpointing.resume_from_ckpt=false \
    "checkpointing.save_dir=${run_dir}" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
    +trainer.check_val_every_n_epoch=null \
    "hydra.run.dir=${run_dir}"
}

run_experiment "${original_gpu}" sudoku_embedding_diffusion original \
  >"${run_root}/original.train.log" 2>&1 &
original_pid=$!
run_experiment "${transposed_gpu}" sudoku_embedding_diffusion_transposed transposed \
  >"${run_root}/transposed.train.log" 2>&1 &
transposed_pid=$!

status=0
wait "${original_pid}" || status=1
wait "${transposed_pid}" || status=1
exit "${status}"

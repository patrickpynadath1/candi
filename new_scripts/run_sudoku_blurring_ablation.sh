#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_sudoku_blurring_ablation.sh [--smoke] GPU_ID [GPU_ID ...]

Examples:
  # Full ablation, evenly queued across four GPUs
  new_scripts/run_sudoku_blurring_ablation.sh 4 5 6 7

  # Real two-step smoke runs of every variant
  new_scripts/run_sudoku_blurring_ablation.sh --smoke 4 5
EOF
}

smoke=false
if [[ ${1:-} == "--smoke" ]]; then
  smoke=true
  shift
fi
if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

gpus=("$@")
for gpu in "${gpus[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU id: ${gpu}" >&2
    usage
    exit 2
  fi
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
worker="${repo_root}/new_scripts/train_sudoku_blurring_one.sh"
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
mode=full
if ${smoke}; then
  mode=smoke
fi
launcher_root="${storage_root}/new_sudoku_runs/blurring_ablation_${mode}"
mkdir -p "${launcher_root}"

# First establish the input bottleneck, then matched isotropic controls, then
# the fixed-codebook diagnostic. Each entry is label:d_in:basis:embedding_mode.
jobs=(
  geometric_reblur_dct_d2:2:dct:live
  geometric_reblur_dct_d4:4:dct:live
  geometric_reblur_dct_d8:8:dct:live
  geometric_reblur_dct_d16:16:dct:live
  geometric_reblur_isotropic_d2:2:isotropic:live
  geometric_reblur_isotropic_d4:4:isotropic:live
  geometric_reblur_isotropic_d8:8:isotropic:live
  geometric_reblur_isotropic_d16:16:isotropic:live
  geometric_reblur_dct_d4_detached:4:dct:detached
)

run_queue() {
  local worker_index=$1
  local gpu_id=${gpus[worker_index]}
  local job_index spec label input_dim basis embedding_mode
  local extra=()
  if ${smoke}; then
    extra=(
      trainer.max_steps=2
      trainer.val_check_interval=2
      data.num_train=64
      data.num_valid=32
      loader.global_batch_size=8
      loader.eval_global_batch_size=8
      loader.num_workers=0
      sampling.steps=2
      callbacks.sudoku_eval.every_n_train_steps=2
      callbacks.sudoku_eval.subset_size=8
      callbacks.sudoku_eval.batch_size=4
      callbacks.sudoku_eval.num_steps=2
      callbacks.checkpoint_every_n_steps.every_n_train_steps=2
      wandb=null
    )
  fi

  for ((job_index=worker_index; job_index<${#jobs[@]}; job_index+=${#gpus[@]})); do
    spec=${jobs[job_index]}
    IFS=: read -r label input_dim basis embedding_mode <<<"${spec}"
    if ${smoke}; then
      label="smoke_${label}"
    fi
    echo "[$(date +%H:%M:%S)] GPU ${gpu_id}: starting ${label}"
    "${worker}" \
      "${gpu_id}" "${label}" "${input_dim}" "${basis}" "${embedding_mode}" \
      "${extra[@]}" \
      >"${launcher_root}/${label}.log" 2>&1
    echo "[$(date +%H:%M:%S)] GPU ${gpu_id}: finished ${label}"
  done
}

pids=()
for index in "${!gpus[@]}"; do
  run_queue "${index}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ ${status} -ne 0 ]]; then
  echo "one or more ablation jobs failed; inspect ${launcher_root}/*.log" >&2
  exit "${status}"
fi
echo "all ${mode} ablation jobs completed; logs: ${launcher_root}"

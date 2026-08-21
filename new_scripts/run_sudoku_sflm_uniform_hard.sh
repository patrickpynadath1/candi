#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_sudoku_sflm_uniform_hard.sh [--smoke] GPU_ID [GPU_ID ...]

Runs the paper-matched truncated+adaptive S-FLM and matched uniform-diffusion
Sudoku-hard jobs. Jobs are assigned round-robin across the supplied GPUs.

Examples:
  new_scripts/run_sudoku_sflm_uniform_hard.sh 0 1
  new_scripts/run_sudoku_sflm_uniform_hard.sh --smoke 0
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
worker="${repo_root}/new_scripts/train_sudoku_sflm_uniform_hard_one.sh"
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
mode=full
${smoke} && mode=smoke
launcher_root="${storage_root}/new_sudoku_runs/sflm_uniform_paper_hard_${mode}"
mkdir -p "${launcher_root}"
jobs=(sflm uniform)

run_queue() {
  local worker_index=$1
  local gpu_id=${gpus[worker_index]}
  local job_index method
  local extra=()
  if ${smoke}; then
    extra=(
      trainer.max_steps=2
      trainer.accumulate_grad_batches=1
      trainer.val_check_interval=2
      trainer.limit_val_batches=1
      data.num_train=32
      data.num_valid=16
      loader.global_batch_size=2
      loader.eval_global_batch_size=2
      loader.num_workers=0
      sampling.steps=2
      callbacks.sudoku_eval.every_n_train_steps=2
      callbacks.sudoku_eval.subset_size=2
      callbacks.sudoku_eval.batch_size=2
      callbacks.sudoku_eval.num_steps=90
      callbacks.sudoku_eval.full_eval_at_end=false
      callbacks.checkpoint_every_n_steps.every_n_train_steps=2
      wandb=null
    )
  fi
  for ((job_index=worker_index; job_index<${#jobs[@]}; job_index+=${#gpus[@]})); do
    method=${jobs[job_index]}
    echo "[$(date +%H:%M:%S)] GPU ${gpu_id}: starting ${method} (${mode})"
    DUO_RUN_MODE="${mode}" "${worker}" "${gpu_id}" "${method}" "${extra[@]}" \
      >"${launcher_root}/${method}.log" 2>&1
    echo "[$(date +%H:%M:%S)] GPU ${gpu_id}: finished ${method} (${mode})"
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
  echo "one or more jobs failed; inspect ${launcher_root}/*.log" >&2
  exit "${status}"
fi
echo "all ${mode} jobs completed; logs: ${launcher_root}"

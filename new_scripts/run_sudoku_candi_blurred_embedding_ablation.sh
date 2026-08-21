#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == "--smoke" ]]; then
  smoke=true
  shift
else
  smoke=false
fi
if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--smoke] GPU_ID [GPU_ID ...]" >&2
  exit 2
fi

gpus=("$@")
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
worker="${repo_root}/new_scripts/train_sudoku_candi_blurred.sh"
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
mode=full
${smoke} && mode=smoke
log_dir="${storage_root}/new_sudoku_runs/candi_blurred_embedding_ablation_${mode}"
mkdir -p "${log_dir}"

# 512 is the normal CANDI input-embedding width in the Sudoku tiny model.
jobs=(d16:16 d512:512)

run_queue() {
  local worker_index=$1 gpu=${gpus[$1]} index spec label dim
  for ((index=worker_index; index<${#jobs[@]}; index+=${#gpus[@]})); do
    spec=${jobs[index]}
    IFS=: read -r label dim <<<"${spec}"
    args=("${gpu}" --input-dim "${dim}" --label "candi_blurred_${label}_hard_eb")
    ${smoke} && args+=(--smoke)
    echo "[$(date +%H:%M:%S)] GPU ${gpu}: starting CANDI_blurred d_in=${dim}"
    "${worker}" "${args[@]}" >"${log_dir}/${label}.log" 2>&1
    echo "[$(date +%H:%M:%S)] GPU ${gpu}: finished CANDI_blurred d_in=${dim}"
  done
}

pids=()
for index in "${!gpus[@]}"; do
  run_queue "${index}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ ${status} -ne 0 ]]; then
  echo "an ablation job failed; inspect ${log_dir}" >&2
  exit "${status}"
fi
echo "CANDI_blurred embedding ablation completed: ${log_dir}"

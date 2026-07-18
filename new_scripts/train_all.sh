#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
algorithms=(mdlm ar mdlm_perm candi cadd duo embedding_diffusion cont_diff_onehot semicat)
gpus=(5 6 7)
pids=()

run_gpu_queue() {
  local gpu=$1
  shift
  local algorithm
  for algorithm in "$@"; do
    "${repo_root}/new_scripts/train_one.sh" "${gpu}" "${algorithm}" \
      >"${repo_root}/new_runs/${algorithm}.train.log" 2>&1
  done
}

mkdir -p "${repo_root}/new_runs"
for index in "${!gpus[@]}"; do
  queue=()
  for ((j=index; j<${#algorithms[@]}; j+=${#gpus[@]})); do
    queue+=("${algorithms[j]}")
  done
  run_gpu_queue "${gpus[index]}" "${queue[@]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"

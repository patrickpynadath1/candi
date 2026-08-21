#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 GPU_ID" >&2; exit 2; fi
gpu=$1
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "invalid GPU id: ${gpu}" >&2; exit 2; }
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
result_root="${storage_root}/new_eval_results/sudoku_hard_eb_0p1_greedy"
worker="${repo_root}/new_scripts/eval_sudoku_hard_eb_greedy_one.sh"
mkdir -p "${result_root}"
jobs=(candi_blurred_d16 candi_blurred_d512 candi mdlm)
for label in "${jobs[@]}"; do
  if [[ -f "${result_root}/${label}/results.json" ]]; then
    echo "skipping completed ${label}"
    continue
  fi
  echo "[$(date +%H:%M:%S)] GPU ${gpu}: loading ${label}"
  "${worker}" "${gpu}" "${label}"
  echo "[$(date +%H:%M:%S)] GPU ${gpu}: completed ${label}"
done
echo "all full evaluations completed: ${result_root}"

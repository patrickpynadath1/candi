#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
algorithms=(mdlm ar candi duo flm sfm)
gpus=(5 6 7)
nfe=${1:-180}
mkdir -p "${repo_root}/new_eval_results/sudoku_adaptive"
pids=()
names=()
status=0

for index in "${!algorithms[@]}"; do
  algorithm=${algorithms[index]}
  gpu=${gpus[index%${#gpus[@]}]}
  "${repo_root}/new_scripts/eval_sudoku_adaptive_one.sh" \
    "${gpu}" "${algorithm}" "${nfe}" \
    >"${repo_root}/new_eval_results/sudoku_adaptive/${algorithm}.eval.log" 2>&1 &
  pids+=("$!")
  names+=("${algorithm}")
done

for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    echo "SUDOKU_ADAPTIVE_EVAL_OK:${names[index]}"
  else
    echo "SUDOKU_ADAPTIVE_EVAL_FAILED:${names[index]}" >&2
    status=1
  fi
done
exit "${status}"

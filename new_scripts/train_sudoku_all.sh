#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
algorithms=(mdlm ar candi duo flm sfm)
gpus=(5 6 7)
mkdir -p "${repo_root}/new_sudoku_runs"

pids=()
names=()
for index in "${!algorithms[@]}"; do
  algorithm=${algorithms[index]}
  gpu=${gpus[index%${#gpus[@]}]}
  "${repo_root}/new_scripts/train_sudoku_one.sh" "${gpu}" "${algorithm}" \
    >"${repo_root}/new_sudoku_runs/${algorithm}.train.log" 2>&1 &
  pids+=("$!")
  names+=("${algorithm}")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    echo "SUDOKU_TRAIN_OK:${names[index]}"
  else
    echo "SUDOKU_TRAIN_FAILED:${names[index]}" >&2
    status=1
  fi
done
exit "${status}"

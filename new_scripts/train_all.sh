#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
algorithms=(mdlm ar mdlm_perm candi cadd duo embedding_diffusion cont_diff_onehot semicat)
gpus=(5 6 7)
mkdir -p "${repo_root}/new_runs"
pids=()
for index in "${!algorithms[@]}"; do
  algorithm=${algorithms[index]}
  gpu=${gpus[index%${#gpus[@]}]}
  "${repo_root}/new_scripts/train_one.sh" "${gpu}" "${algorithm}" \
    >"${repo_root}/new_runs/${algorithm}.train.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"

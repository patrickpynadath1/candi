#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
algorithms=(mdlm ar mdlm_perm candi cadd duo cont_diff_embed cont_diff_onehot semicat)
nfes=(16 32 64 128)
temperatures=(0.7 0.85 1.0 1.15 1.3)
gpus=(5 6 7)

mkdir -p "${repo_root}/new_results"
run_index=0
pids=()
for algorithm in "${algorithms[@]}"; do
  current_nfes=("${nfes[@]}")
  if [[ "${algorithm}" == "ar" || "${algorithm}" == "mdlm_perm" ]]; then
    current_nfes=(128)
  fi
  for nfe in "${current_nfes[@]}"; do
    for temperature in "${temperatures[@]}"; do
      gpu=${gpus[run_index%${#gpus[@]}]}
      "${repo_root}/new_scripts/eval_one.sh" "${gpu}" "${algorithm}" "${nfe}" "${temperature}" \
        >"${repo_root}/new_results/${algorithm}_nfe_${nfe}_temp_${temperature}.log" 2>&1 &
      pids+=("$!")
      run_index=$((run_index + 1))
      if [[ ${#pids[@]} -eq ${#gpus[@]} ]]; then
        for pid in "${pids[@]}"; do wait "${pid}"; done
        pids=()
      fi
    done
  done
done
for pid in "${pids[@]}"; do wait "${pid}"; done

"${repo_root}/.venv/bin/python" "${repo_root}/new_scripts/aggregate_frontiers.py"

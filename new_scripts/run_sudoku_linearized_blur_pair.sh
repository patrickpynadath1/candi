#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_sudoku_linearized_blur_pair.sh [OPTIONS] --gpus ID[,ID...]

Options:
  --smoke                 Two-step training smoke test with real sampling
  --gpus 0,1              GPU IDs; one GPU runs the two jobs sequentially
Examples:
  new_scripts/run_sudoku_linearized_blur_pair.sh --smoke --gpus 0
  new_scripts/run_sudoku_linearized_blur_pair.sh --gpus 0,1
EOF
}

smoke=false
gpus=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) smoke=true; shift ;;
    --gpus) IFS=',' read -r -a gpus <<<"$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
if [[ ${#gpus[@]} -eq 0 ]]; then
  usage
  exit 2
fi
for gpu in "${gpus[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "invalid GPU id: ${gpu}" >&2; exit 2; }
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
worker="${repo_root}/new_scripts/train_sudoku_linearized_blur_one.sh"
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
mode=full
${smoke} && mode=smoke
log_dir="${storage_root}/new_sudoku_runs/linearized_blur_pair_${mode}_logs"
mkdir -p "${log_dir}"
# The smaller corruption tables did not improve the earlier Sudoku results.
# Keep both methods matched at the largest tested embedding width.
jobs=("pure:512" "candi:512")

run_queue() {
  local slot=$1 gpu=${gpus[$1]} index spec variant dim
  for ((index=slot; index<${#jobs[@]}; index+=${#gpus[@]})); do
    spec=${jobs[index]}
    IFS=: read -r variant dim <<<"${spec}"
    args=("${gpu}" "${variant}" --input-dim "${dim}")
    ${smoke} && args+=(--smoke)
    echo "[$(date +%H:%M:%S)] GPU ${gpu}: starting ${variant} linearized blur d${dim}"
    "${worker}" "${args[@]}" >"${log_dir}/${variant}_d${dim}.log" 2>&1
    echo "[$(date +%H:%M:%S)] GPU ${gpu}: finished ${variant} linearized blur d${dim}"
  done
}

pids=()
for slot in "${!gpus[@]}"; do
  run_queue "${slot}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ ${status} -ne 0 ]]; then
  echo "a job failed; inspect ${log_dir}" >&2
  exit "${status}"
fi
echo "linearized blur pair completed: ${log_dir}"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_sudoku_two_axis_rho_sweep.sh [--smoke] --gpus ID[,ID...]

Runs six matched d_in=512 jobs on the geometric (unwarped) schedule:
  pure continuous: rho = 0.25, 0.5, 0.75
  hybrid CANDI:    rho = 0.25, 0.5, 0.75

Examples:
  new_scripts/run_sudoku_two_axis_rho_sweep.sh --smoke --gpus 0,1
  new_scripts/run_sudoku_two_axis_rho_sweep.sh --gpus 2,3,6,7
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
log_dir="${storage_root}/new_sudoku_runs/two_axis_rho_sweep_${mode}_logs"
mkdir -p "${log_dir}"

jobs=(
  pure:0.25
  pure:0.5
  pure:0.75
  candi:0.25
  candi:0.5
  candi:0.75
)

run_queue() {
  local slot=$1 gpu=${gpus[$1]} index spec variant rho
  for ((index=slot; index<${#jobs[@]}; index+=${#gpus[@]})); do
    spec=${jobs[index]}
    IFS=: read -r variant rho <<<"${spec}"
    args=(
      "${gpu}" "${variant}"
      --input-dim 512
      --rho "${rho}"
      --schedule geometric
    )
    ${smoke} && args+=(--smoke)
    echo "[$(date +%H:%M:%S)] GPU ${gpu}: starting ${variant} d512 rho=${rho} geometric"
    "${worker}" "${args[@]}" >"${log_dir}/${variant}_rho${rho}_d512.log" 2>&1
    echo "[$(date +%H:%M:%S)] GPU ${gpu}: finished ${variant} d512 rho=${rho} geometric"
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
echo "two-axis rho sweep completed: ${log_dir}"

#!/usr/bin/env bash
set -euo pipefail

smoke=false
temperature=1.0
gpus=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) smoke=true; shift ;;
    --temperature) temperature=$2; shift 2 ;;
    --gpus) IFS=',' read -r -a gpus <<<"$2"; shift 2 ;;
    *) gpus+=("$1"); shift ;;
  esac
done
if [[ ${#gpus[@]} -eq 0 ]]; then
  echo "usage: $0 [--smoke] [--temperature T] --gpus 0,1" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
worker="${repo_root}/new_scripts/train_text8_candi_blurred_one.sh"
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
mode=full
${smoke} && mode=smoke
log_dir="${storage_root}/new_runs/text8_candi_blurred_ablation_${mode}_logs"
mkdir -p "${log_dir}"
jobs=(16 256)

run_queue() {
  local slot=$1 gpu=${gpus[$1]} index dim
  for ((index=slot; index<${#jobs[@]}; index+=${#gpus[@]})); do
    dim=${jobs[index]}
    args=("${gpu}" "${dim}" --temperature "${temperature}")
    ${smoke} && args+=(--smoke)
    echo "[$(date +%H:%M:%S)] GPU ${gpu}: starting Text8 CANDI_blurred d${dim}"
    "${worker}" "${args[@]}" >"${log_dir}/d${dim}.log" 2>&1
    echo "[$(date +%H:%M:%S)] GPU ${gpu}: finished Text8 CANDI_blurred d${dim}"
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
echo "Text8 CANDI_blurred ablation completed: ${log_dir}"

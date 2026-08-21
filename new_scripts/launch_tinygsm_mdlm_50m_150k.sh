#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 GPU_ID [RUN_LABEL]" >&2
  exit 2
fi

repo_root="/home/patrick/duo"
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
gpu_id=$1
run_label=${2:-mdlm50m-lr3e-4-b64}
worker="${repo_root}/new_scripts/train_tinygsm_mdlm_50m_150k.sh"
output_dir="${storage_root}/new_training_runs/tinygsm_mdlm_50m/${run_label}"
log_dir="${output_dir}/launcher"
pid_path="${log_dir}/train.pid"
log_path="${log_dir}/train.log"
mkdir -p "${log_dir}"

if [[ -f "${pid_path}" ]]; then
  old_pid=$(<"${pid_path}")
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "already running: PID ${old_pid}" >&2
    exit 1
  fi
fi

nohup setsid "${worker}" "${gpu_id}" "${run_label}" \
  </dev/null >"${log_path}" 2>&1 &
pid=$!
echo "${pid}" >"${pid_path}"
echo "launched ${run_label}: GPU=${gpu_id} PID=${pid} log=${log_path}"

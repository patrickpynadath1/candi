#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "usage: $0 [MDLM_GPU] [CANDI_GPU]" >&2
  exit 2
fi

mdlm_gpu=${1:-6}
candi_gpu=${2:-7}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
worker="${repo_root}/new_scripts/train_sudoku_hard_one.sh"
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}

launch_one() {
  local gpu_id=$1
  local algorithm=$2
  local run_dir="${storage_root}/new_sudoku_runs/hard_eb_0p1/${algorithm}"
  local launcher_dir="${run_dir}/launcher"
  local pid_path="${launcher_dir}/train.pid"
  local log_path="${launcher_dir}/train.log"
  mkdir -p "${launcher_dir}"
  if [[ -f "${pid_path}" ]]; then
    local old_pid
    old_pid=$(<"${pid_path}")
    if kill -0 "${old_pid}" 2>/dev/null; then
      echo "${algorithm} already running: PID=${old_pid}" >&2
      return 1
    fi
  fi
  nohup setsid "${worker}" "${gpu_id}" "${algorithm}" \
    </dev/null >"${log_path}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${pid_path}"
  echo "launched ${algorithm}: GPU=${gpu_id} PID=${pid} log=${log_path}"
}

launch_one "${mdlm_gpu}" mdlm
launch_one "${candi_gpu}" candi

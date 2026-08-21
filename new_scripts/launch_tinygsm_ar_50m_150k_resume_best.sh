#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/patrick/duo"
worker="${repo_root}/new_scripts/train_tinygsm_ar_50m_150k_resume_best.sh"
log_dir="${repo_root}/new_smoke_runs/tinygsm_ar_50m_2h/launcher_logs"
mkdir -p "${log_dir}"

launch_run() {
  local gpu_id=$1
  local learning_rate=$2
  local run_label=$3
  local wandb_run_id=$4
  local checkpoint="${repo_root}/new_smoke_runs/tinygsm_ar_50m_2h/${run_label}/checkpoints/best.ckpt"
  local log_path="${log_dir}/${run_label}.log"
  local pid_path="${log_dir}/${run_label}.pid"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    return 1
  fi

  if [[ -f "${pid_path}" ]]; then
    local old_pid
    old_pid=$(<"${pid_path}")
    if kill -0 "${old_pid}" 2>/dev/null; then
      echo "already running: ${run_label} (PID ${old_pid})" >&2
      return 1
    fi
  fi

  nohup setsid "${worker}" "${gpu_id}" "${learning_rate}" "${run_label}" "${wandb_run_id}" \
    </dev/null >"${log_path}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${pid_path}"
  echo "launched ${run_label}: GPU=${gpu_id} PID=${pid} log=${log_path}"
}

launch_run 5 3e-4 ar50m-lr3e-4-b64 7hzc6ijo
launch_run 6 6e-4 ar50m-lr6e-4-b64 xpgw39r1
launch_run 7 1e-3 ar50m-lr1e-3-b64 3qtgn9ib

echo
echo "Monitor with:"
echo "  tail -F ${log_dir}/ar50m-*.log"
echo "  nvidia-smi"

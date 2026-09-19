#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 OUTPUT_ROOT WARMUP_GPU DIRECT_GPU GPU_1K GPU_5K GPU_10K" >&2
  exit 2
fi

output_root=$(realpath -m "$1")
warmup_gpu=$2
direct_gpu=$3
gpu_1k=$4
gpu_5k=$5
gpu_10k=$6
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "${output_root}"

run_main() {
  local gpu=$1
  local directory=$2
  shift 2
  mkdir -p "${directory}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${repo_root}/.venv/bin/python" \
    "${repo_root}/main.py" "$@" \
    trainer.devices=1 \
    checkpointing.save_dir="${directory}" \
    hydra.run.dir="${directory}" \
    >"${directory}/launcher.log" 2>&1
}

warmup_dir="${output_root}/warmup"
run_main "${warmup_gpu}" "${warmup_dir}" \
  +experiment=sudoku_extreme_frm_warmup_milestones \
  wandb.name=frm-warmup-10k &
warmup_pid=$!

run_main "${direct_gpu}" "${output_root}/fpf_0k" \
  +experiment=sudoku_extreme_frm_fpf_pilot \
  wandb.name=frm-fpf-from-0k &
pids=($!)

launch_at_milestone() {
  local step=$1
  local gpu=$2
  local checkpoint="${warmup_dir}/checkpoints/step_$(printf '%06d' "${step}").ckpt"
  local label="${step}"
  while [[ ! -f "${checkpoint}" ]]; do
    if ! kill -0 "${warmup_pid}" 2>/dev/null; then
      echo "warmup exited before producing ${checkpoint}" >&2
      return 1
    fi
    sleep 10
  done
  echo "launching FPF after ${step} warmup steps on GPU ${gpu}"
  run_main "${gpu}" "${output_root}/fpf_${label}" \
    +experiment=sudoku_extreme_frm_fpf_pilot \
    training.finetune_path="${checkpoint}" \
    wandb.name="frm-fpf-after-${label}" &
  pids+=($!)
}

launch_at_milestone 1000 "${gpu_1k}"
launch_at_milestone 5000 "${gpu_5k}"
launch_at_milestone 10000 "${gpu_10k}"

wait "${warmup_pid}"
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ ${status} -ne 0 ]]; then
  echo "one or more FPF pilots failed; inspect ${output_root}/*/launcher.log" >&2
  exit "${status}"
fi
echo "all FPF warmup-ablation runs completed under ${output_root}"

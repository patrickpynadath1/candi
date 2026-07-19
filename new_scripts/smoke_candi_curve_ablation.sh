#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="/home/patrick/duo/.venv/bin/python"
root="${repo_root}/new_candi_curve_runs/smoke"
mkdir -p "${root}"

specs=(
  "linear|sudoku_candi_curve_5k|algo.curve_type=linear"
  "power|sudoku_candi_curve_5k|algo.curve_type=power algo.curve_orientation=continuous_first algo.curve_power=4"
  "bezier|sudoku_candi_curve_5k|algo.curve_type=bezier algo.curve_control_1=[0.05,0.55] algo.curve_control_2=[0.45,0.95]"
  "adaptive|sudoku_candi_curve_adaptive_5k|algo.curve_type=power algo.curve_orientation=continuous_first algo.curve_power=4 noise.adaptive_buffer_size=2 noise.adaptive_refit_every=1 noise.adaptive_warmup_steps=0 noise.adaptive_n_grid=32 noise.adaptive_n_knots=4 noise.adaptive_spline_degree=2"
  "factorized|sudoku_candi_factorized_probe_5k|noise.adaptive_buffer_size=2 noise.adaptive_refit_every=1 noise.adaptive_warmup_steps=0 noise.adaptive_n_grid=32 noise.adaptive_n_knots=4 noise.adaptive_spline_degree=2 algo.joint_loss_bins=4 algo.joint_loss_log_every=1"
)
gpus=(5 6 7)
pids=()
names=()
status=0
for index in "${!specs[@]}"; do
  IFS='|' read -r name experiment overrides <<<"${specs[index]}"
  read -r -a override_args <<<"${overrides}"
  run_dir="${root}/${name}"
  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpus[index%3]}" "${python_bin}" "${repo_root}/main.py" \
    --config-name=config "+experiment=${experiment}" \
    +trainer.fast_dev_run=true trainer.accumulate_grad_batches=1 \
    loader.global_batch_size=2 wandb=null \
    data.cache_dir=/home/patrick/duo/data_cache \
    "checkpointing.save_dir=${run_dir}" "hydra.run.dir=${run_dir}" \
    "${override_args[@]}" >"${run_dir}/smoke.log" 2>&1 &
  pids+=("$!")
  names+=("${name}")
done
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    echo "CANDI_CURVE_SMOKE_OK:${names[index]}"
  else
    echo "CANDI_CURVE_SMOKE_FAILED:${names[index]}" >&2
    status=1
  fi
done
exit "${status}"

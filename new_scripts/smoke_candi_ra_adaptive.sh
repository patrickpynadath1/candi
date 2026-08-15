#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="${repo_root}/.venv/bin/python"
smoke_root="${repo_root}/new_smoke_runs/candi_ra_adaptive"
gpus=(5 6 7)
specs=(
  "fixed|sudoku_candi_ra_change"
  "scalar_adaptive|sudoku_candi_ra_adaptive"
  "factorized_adaptive|sudoku_candi_ra_factorized_adaptive"
)

mkdir -p "${smoke_root}"
pids=()
labels=()
for index in "${!specs[@]}"; do
  IFS='|' read -r label experiment <<<"${specs[index]}"
  run_dir="${smoke_root}/${label}"
  mkdir -p "${run_dir}"
  adaptive_overrides=()
  if [[ "${label}" != "fixed" ]]; then
    adaptive_overrides=(
      noise.adaptive_buffer_size=2 noise.adaptive_refit_every=1
      noise.adaptive_warmup_steps=0 noise.adaptive_n_grid=32
      noise.adaptive_n_knots=4 noise.adaptive_spline_degree=2
    )
  fi
  CUDA_VISIBLE_DEVICES="${gpus[index]}" "${python_bin}" "${repo_root}/main.py" \
    --config-name=config "+experiment=${experiment}" \
    +trainer.fast_dev_run=true trainer.accumulate_grad_batches=1 \
    loader.global_batch_size=2 wandb=null \
    algo.joint_loss_bins=4 algo.joint_loss_log_every=1 \
    data.cache_dir=/home/patrick/duo/data_cache \
    "checkpointing.save_dir=${run_dir}" "hydra.run.dir=${run_dir}" \
    "${adaptive_overrides[@]}" \
    >"${run_dir}/smoke.log" 2>&1 &
  pids+=("$!")
  labels+=("${label}")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    echo "CANDI_RA_ADAPTIVE_SMOKE_OK:${labels[index]}"
  else
    echo "CANDI_RA_ADAPTIVE_SMOKE_FAILED:${labels[index]}" >&2
    status=1
  fi
done
exit "${status}"

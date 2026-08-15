#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bandwidths=(0.0 0.5 1.5)
gpus=(1 2 3)
sweep_root="${repo_root}/new_sudoku_runs/candi_ra_5k_wandb"
mkdir -p "${sweep_root}"

pids=()
labels=()
for index in "${!bandwidths[@]}"; do
  bandwidth=${bandwidths[index]}
  gpu=${gpus[index]}
  label="bandwidth_${bandwidth}"
  wandb_label=${bandwidth/./p}
  run_dir="${sweep_root}/${label}"
  mkdir -p "${run_dir}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
    --config-name=config \
    +experiment=sudoku_candi_ra_change \
    sampling.predictor=candi \
    "algo.spatial_blurring.bandwidth=${bandwidth}" \
    algo.min_percentile=0.05 \
    algo.max_percentile=0.4 \
    wandb.project=sudoku-candi-ra-spatial-5k \
    wandb.group=gaussian-bandwidth-sweep-p05-p4 \
    "wandb.name=candi-ra-sudoku-5k-bw-${wandb_label}" \
    wandb.job_type=train \
    checkpointing.resume_from_ckpt=false \
    "checkpointing.save_dir=${run_dir}" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
    "hydra.run.dir=${run_dir}" \
    >"${run_dir}/train.log" 2>&1 &
  pids+=("$!")
  labels+=("${label}")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    echo "SUDOKU_CANDI_RA_5K_OK:${labels[index]}"
  else
    echo "SUDOKU_CANDI_RA_5K_FAILED:${labels[index]}" >&2
    status=1
  fi
done
exit "${status}"

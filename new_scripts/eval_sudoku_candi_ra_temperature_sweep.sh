#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bandwidths=(0.0 0.5 1.5)
temperatures=(0.05)
gpus=(1 2 3)
nfe=${1:-180}
train_root="${repo_root}/new_sudoku_runs/candi_ra_5k_wandb"
eval_root="${repo_root}/new_eval_results/sudoku_candi_ra_5k_temperature/nfe_${nfe}"
mkdir -p "${eval_root}"

run_bandwidth() {
  local bandwidth=$1
  local gpu=$2
  local label="bandwidth_${bandwidth}"
  local checkpoint="${train_root}/${label}/checkpoints/last.ckpt"
  local temperature output_dir
  for temperature in "${temperatures[@]}"; do
    output_dir="${eval_root}/${label}/temp_${temperature}"
    mkdir -p "${output_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
      --config-name=config \
      +experiment=sudoku_candi_ra_change \
      mode=sudoku_eval \
      sampling.predictor=candi \
      "sampling.steps=${nfe}" \
      "algo.spatial_blurring.bandwidth=${bandwidth}" \
      "algo.temp=${temperature}" \
      algo.min_percentile=0.05 \
      algo.max_percentile=0.4 \
      sudoku.batch_size=64 \
      loader.eval_global_batch_size=64 \
      loader.num_workers=4 \
      "eval.checkpoint_path=${checkpoint}" \
      "sudoku.output_dir=${output_dir}" \
      "hydra.run.dir=${output_dir}" \
      wandb=null \
      >"${output_dir}/eval.log" 2>&1
    echo "SUDOKU_TEMP_EVAL_OK:${label}:temp_${temperature}"
  done
}

pids=()
for index in "${!bandwidths[@]}"; do
  run_bandwidth "${bandwidths[index]}" "${gpus[index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"

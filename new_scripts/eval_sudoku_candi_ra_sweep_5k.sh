#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bandwidths=(0.0 0.5 1.5)
gpus=(1 2 3)
nfe=${1:-180}
train_root="${repo_root}/new_sudoku_runs/candi_ra_5k_wandb"
eval_root="${repo_root}/new_eval_results/sudoku_candi_ra_5k/nfe_${nfe}"
mkdir -p "${eval_root}"

pids=()
labels=()
for index in "${!bandwidths[@]}"; do
  bandwidth=${bandwidths[index]}
  gpu=${gpus[index]}
  label="bandwidth_${bandwidth}"
  checkpoint="${train_root}/${label}/checkpoints/last.ckpt"
  output_dir="${eval_root}/${label}"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  mkdir -p "${output_dir}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
    --config-name=config \
    +experiment=sudoku_candi_ra_change \
    mode=sudoku_eval \
    sampling.predictor=candi \
    "sampling.steps=${nfe}" \
    "algo.spatial_blurring.bandwidth=${bandwidth}" \
    algo.min_percentile=0.05 \
    algo.max_percentile=0.4 \
    sudoku.batch_size=64 \
    loader.eval_global_batch_size=64 \
    loader.num_workers=4 \
    "eval.checkpoint_path=${checkpoint}" \
    "sudoku.output_dir=${output_dir}" \
    "hydra.run.dir=${output_dir}" \
    wandb=null \
    >"${output_dir}/eval.log" 2>&1 &
  pids+=("$!")
  labels+=("${label}")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    echo "SUDOKU_CANDI_RA_EVAL_OK:${labels[index]}"
  else
    echo "SUDOKU_CANDI_RA_EVAL_FAILED:${labels[index]}" >&2
    status=1
  fi
done
exit "${status}"

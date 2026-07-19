#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU_ID RUN_NAME EXPERIMENT [OVERRIDES ...]" >&2
  exit 2
fi
gpu_id=$1
run_name=$2
experiment=$3
shift 3
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="/home/patrick/duo/.venv/bin/python"
checkpoint="${repo_root}/new_candi_curve_runs/${run_name}/checkpoints/last.ckpt"
nfe=${CANDI_ABLATION_NFE:-90}
num_valid=${CANDI_ABLATION_NUM_VALID:-500}
output_dir="${repo_root}/new_candi_curve_results/${run_name}/nfe_${nfe}"
mkdir -p "${output_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" "${repo_root}/main.py" \
  --config-name=config "+experiment=${experiment}" mode=sudoku_eval \
  sampling.predictor=candi "sampling.steps=${nfe}" \
  "data.num_valid=${num_valid}" data.cache_dir=/home/patrick/duo/data_cache \
  sudoku.batch_size=64 loader.eval_global_batch_size=64 loader.num_workers=4 \
  "eval.checkpoint_path=${checkpoint}" "sudoku.output_dir=${output_dir}" \
  "hydra.run.dir=${output_dir}" wandb=null "$@"

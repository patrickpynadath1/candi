#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 GPU_ID ALGORITHM [NFE]" >&2
  exit 2
fi

gpu_id=$1
algorithm=$2
nfe=${3:-180}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint="${repo_root}/new_sudoku_runs/${algorithm}/checkpoints/last.ckpt"
output_dir="${repo_root}/new_eval_results/sudoku/${algorithm}/nfe_${nfe}"

if [[ ! -f "${checkpoint}" ]]; then
  echo "missing checkpoint: ${checkpoint}" >&2
  exit 1
fi

case "${algorithm}" in
  mdlm) predictor=ancestral_cache ;;
  duo) predictor=ancestral ;;
  ar) predictor=ar ;;
  candi) predictor=candi ;;
  flm) predictor=flm_euler ;;
  sfm) predictor=sfm ;;
  *) echo "unknown Sudoku algorithm: ${algorithm}" >&2; exit 2 ;;
esac

mkdir -p "${output_dir}"
extra=()
if [[ "${algorithm}" == "sfm" ]]; then
  extra+=(
    noise.eps=1e-3
    sampling.noise_removal=greedy
    +sampling.velocity=exact
    +sampling.top_k_velocity=-1
  )
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config \
  "+experiment=sudoku_${algorithm}" \
  mode=sudoku_eval \
  "sampling.predictor=${predictor}" \
  "sampling.steps=${nfe}" \
  sudoku.batch_size=64 \
  loader.eval_global_batch_size=64 \
  loader.num_workers=4 \
  "eval.checkpoint_path=${checkpoint}" \
  "sudoku.output_dir=${output_dir}" \
  "hydra.run.dir=${output_dir}" \
  wandb=null \
  "${extra[@]}"

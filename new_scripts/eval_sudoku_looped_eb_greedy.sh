#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 GPU_ID ALGORITHM CHECKPOINT OUTPUT_DIR [HYDRA_OVERRIDES...]" >&2
  exit 2
fi
gpu_id=$1
algorithm=$2
checkpoint=$3
output_dir=$4
shift 4
case "${algorithm}" in
  candi_looped|candi_register_looped|candi_softmax_looped) ;;
  *) echo "unsupported looped algorithm: ${algorithm}" >&2; exit 2 ;;
esac
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "${output_dir}"
cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
exec "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  "+experiment=sudoku_${algorithm}" \
  mode=sudoku_eval data.difficulty=hard \
  "data.cache_dir=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}/data_cache" \
  "eval.checkpoint_path=${checkpoint}" \
  eval.disable_ema=false \
  sampling.predictor=candi_looped_entropy_bounded \
  sampling.entropy_bound=0.1 sampling.greedy_clean_estimate=true \
  sampling.steps=180 \
  sudoku.batch_size=64 \
  "sudoku.output_dir=${output_dir}" "hydra.run.dir=${output_dir}" \
  "$@"

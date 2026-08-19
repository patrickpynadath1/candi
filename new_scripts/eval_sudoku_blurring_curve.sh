#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 GPU_ID LABEL D_IN [dct|isotropic]" >&2
  exit 2
fi

gpu_id=$1
label=$2
input_dim=$3
basis=${4:-dct}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
run_dir="${storage_root}/new_sudoku_runs/blurring/${label}"
checkpoint="${run_dir}/checkpoints/last.ckpt"

if [[ ! -f "${checkpoint}" ]]; then
  echo "missing checkpoint: ${checkpoint}" >&2
  exit 1
fi

for nfe in 1 2 4 8 16 32 64 128 256; do
  output_dir="${run_dir}/evaluation/nfe_${nfe}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
    --config-name=config \
    +experiment=sudoku_blurring_hard \
    mode=sudoku_eval \
    "algo.input_embedding_dim=${input_dim}" \
    "algo.basis=${basis}" \
    "sampling.steps=${nfe}" \
    sudoku.batch_size=64 \
    loader.eval_global_batch_size=64 \
    "eval.checkpoint_path=${checkpoint}" \
    "sudoku.output_dir=${output_dir}" \
    "hydra.run.dir=${output_dir}" \
    wandb=null
done

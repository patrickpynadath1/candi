#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 GPU_ID OUTPUT_ROOT [SEED]" >&2
  exit 2
fi
if [[ ! $1 =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer" >&2
  exit 2
fi

gpu_id=$1
output_root=$(realpath -m "$2")
seed=${3:-1}
if [[ ! $seed =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer" >&2
  exit 2
fi
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stage_a="${output_root}/stage_a"
stage_b="${output_root}/stage_b"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  +experiment=sudoku_extreme_frm_stage_a \
  seed="${seed}" \
  trainer.devices=1 \
  checkpointing.save_dir="${stage_a}" \
  hydra.run.dir="${stage_a}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  +experiment=sudoku_extreme_frm_stage_b \
  seed="${seed}" \
  trainer.devices=1 \
  training.finetune_path="${stage_a}/checkpoints/last.ckpt" \
  checkpointing.save_dir="${stage_b}" \
  hydra.run.dir="${stage_b}"

echo "FRM training complete: ${stage_b}/checkpoints/last.ckpt"

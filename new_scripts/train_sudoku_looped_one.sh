#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU_ID ALGORITHM RUN_DIR [HYDRA_OVERRIDES...]" >&2
  exit 2
fi
gpu_id=$1
algorithm=$2
run_dir=$3
shift 3
case "${algorithm}" in
  candi_looped|candi_register_looped|candi_softmax_looped) ;;
  *) echo "unsupported looped algorithm: ${algorithm}" >&2; exit 2 ;;
esac
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "${run_dir}"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
exec "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  "+experiment=sudoku_${algorithm}" \
  data.difficulty=hard \
  "data.cache_dir=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}/data_cache" \
  'callbacks=[checkpoint_every_n_steps,checkpoint_monitor,learning_rate_monitor,sudoku_eval]' \
  trainer.devices=1 \
  loader.num_workers=0 \
  +trainer.check_val_every_n_epoch=null \
  +trainer.enable_progress_bar=false \
  trainer.val_check_interval=4000 \
  wandb.project=sudoku-looping-20k \
  "wandb.name=${algorithm}-sudoku-hard-20k-seed1" \
  wandb.group=looping-hard-seed1 \
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=${run_dir}" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  callbacks.sudoku_eval.every_n_train_steps=1000 \
  callbacks.sudoku_eval.full_eval_at_end=true \
  "callbacks.sudoku_eval.output_dir=${run_dir}/sudoku_eval" \
  "hydra.run.dir=${run_dir}" \
  "$@"

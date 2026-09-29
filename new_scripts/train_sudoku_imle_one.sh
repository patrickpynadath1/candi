#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU_ID K RUN_DIR [HYDRA_OVERRIDES...]" >&2
  exit 2
fi
gpu_id=$1
k=$2
run_dir=$3
shift 3
case "${k}" in
  4|8|16|32) ;;
  *) echo "unsupported candidate count: ${k}" >&2; exit 2 ;;
esac
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "${run_dir}"
run_dir=$(cd "${run_dir}" && pwd)
cd "${repo_root}"

# Hold candidate activations per microbatch and effective batch size fixed.
accumulation=$((2 * k))
microbatch=$((128 / k))
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
exec "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  +experiment=sudoku_candi_imle \
  "algo.imle_candidates=${k}" \
  data.difficulty=hard \
  "data.cache_dir=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}/data_cache" \
  seed=1 trainer.devices=1 trainer.max_steps=20000 \
  loader.num_workers=0 loader.global_batch_size=256 \
  "loader.batch_size=${microbatch}" loader.eval_batch_size=64 \
  "trainer.accumulate_grad_batches=${accumulation}" \
  "trainer.val_check_interval=$((1000 * accumulation))" \
  +trainer.check_val_every_n_epoch=null \
  +trainer.enable_progress_bar=false \
  wandb.project=sudoku-imle-hard-20k \
  "wandb.name=candi-imle-k${k}-sudoku-hard-20k-seed1" \
  wandb.group=imle-hard-k-sweep-seed1 \
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=${run_dir}" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  callbacks.sudoku_eval.every_n_train_steps=1000 \
  callbacks.sudoku_eval.full_eval_at_end=true \
  "callbacks.sudoku_eval.output_dir=${run_dir}/sudoku_eval" \
  "hydra.run.dir=${run_dir}" \
  "$@"

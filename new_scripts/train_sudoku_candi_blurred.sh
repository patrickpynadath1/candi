#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 GPU_ID [--smoke] [--input-dim N] [--label NAME]" >&2
  exit 2
fi

gpu_id=$1
shift
smoke=false
input_dim=16
label=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) smoke=true; shift ;;
    --input-dim) input_dim=$2; shift 2 ;;
    --label) label=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
if [[ -z ${label} ]]; then
  label="candi_blurred_d${input_dim}_hard_eb"
fi
extra=()
if ${smoke}; then
  label="smoke_${label}"
  extra=(
    trainer.max_steps=2
    trainer.val_check_interval=2
    data.num_train=64
    data.num_valid=32
    loader.global_batch_size=8
    loader.eval_global_batch_size=8
    loader.num_workers=0
    callbacks.sudoku_eval.every_n_train_steps=2
    callbacks.sudoku_eval.subset_size=8
    callbacks.sudoku_eval.batch_size=4
    callbacks.sudoku_eval.num_steps=180
    callbacks.checkpoint_every_n_steps.every_n_train_steps=2
    wandb=null
  )
fi
run_dir="${storage_root}/new_sudoku_runs/${label}"
mkdir -p "${run_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config \
  +experiment=sudoku_candi_blurred_hard_eb \
  "data.cache_dir=${storage_root}/data_cache" \
  "algo.input_embedding_dim=${input_dim}" \
  wandb.project=sudoku-blurring-hard-20k \
  wandb.group=candi-blurred-embedding-ablation \
  "wandb.name=${label}" \
  "wandb.tags=[sudoku,hard,candi-blurred,entropy-bounded,argmax,din-${input_dim}]" \
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=${run_dir}" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  +trainer.check_val_every_n_epoch=null \
  "hydra.run.dir=${run_dir}" \
  "${extra[@]}"

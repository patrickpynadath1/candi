#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 GPU_ID [RUN_GROUP]" >&2
  exit 2
fi

repo_root="/home/patrick/duo"
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
gpu_id=$1
run_group=${2:-sudoku-hard-replaid-4way-20k}
group_dir="${storage_root}/new_sudoku_runs/${run_group}"
dataset_cache="${storage_root}/data_cache"

mkdir -p "${group_dir}"
cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

variants=(
  replaid
  replaid_keep_mse
  replaid_keep_hybrid_elbo
  replaid_keep_mse_reveal_ce
)
experiments=(
  sudoku_replaid_hard
  sudoku_replaid_keep_mse_hard_eb
  sudoku_replaid_keep_hybrid_elbo_hard_eb
  sudoku_replaid_keep_mse_reveal_ce_hard_eb
)

for index in "${!variants[@]}"; do
  variant=${variants[index]}
  experiment=${experiments[index]}
  output_dir="${group_dir}/${variant}"
  mkdir -p "${output_dir}"
  echo "[$(date --iso-8601=seconds)] starting ${variant} on CUDA device ${gpu_id}"

  "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
    --config-name=config \
    +experiment="${experiment}" \
    data.cache_dir="${dataset_cache}" \
    trainer.max_steps=20000 \
    trainer.val_check_interval=20000 \
    trainer.devices=1 \
    trainer.accumulate_grad_batches=1 \
    wandb.project=sudoku-hard-replaid-20k \
    wandb.group="${run_group}" \
    wandb.name="${run_group}-${variant}" \
    wandb.tags="[sudoku,hard,${variant},replaid-embedding,20k-steps]" \
    checkpointing.resume_from_ckpt=false \
    checkpointing.save_dir="${output_dir}" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
    +trainer.check_val_every_n_epoch=null \
    hydra.run.dir="${output_dir}"

  echo "[$(date --iso-8601=seconds)] completed ${variant}"
done

echo "[$(date --iso-8601=seconds)] completed all variants"

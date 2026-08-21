#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 GPU_ID [RUN_GROUP]" >&2
  exit 2
fi

repo_root="/home/patrick/duo"
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
gpu_id=$1
run_group=${2:-replaid-keep-50m-20k}
group_dir="${storage_root}/new_training_runs/tinygsm_replaid_keep_50m/${run_group}"

mkdir -p "${group_dir}"
cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

variants=(
  replaid_keep_mse
  replaid_keep_hybrid_elbo
  replaid_keep_mse_reveal_ce
)

for variant in "${variants[@]}"; do
  experiment="tinygsm_${variant}_50m"
  output_dir="${group_dir}/${variant}"
  mkdir -p "${output_dir}"
  echo "[$(date --iso-8601=seconds)] starting ${variant} on CUDA device ${gpu_id}"

  uv run python main.py \
    +experiment="${experiment}" \
    scratch_dir="${output_dir}" \
    checkpointing.save_dir="${output_dir}" \
    checkpointing.resume_from_ckpt=false \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=2000 \
    trainer.devices=1 \
    trainer.accumulate_grad_batches=1 \
    trainer.max_steps=20000 \
    callbacks.gsm8k_eval.every_n_train_steps=2000 \
    callbacks.gsm8k_eval.subset_size=128 \
    callbacks.gsm8k_eval.batch_size=16 \
    callbacks.gsm8k_eval.num_steps=32 \
    callbacks.gsm8k_eval.log_examples=16 \
    callbacks.gsm8k_eval.full_eval_at_end=false \
    wandb.project=tinygsm-replaid-keep-50m \
    wandb.group="${run_group}" \
    wandb.name="${run_group}-${variant}" \
    wandb.tags="[tinygsm,replaid-keep,${variant},50m,context512,20k-steps]"

  echo "[$(date --iso-8601=seconds)] completed ${variant}"
done

echo "[$(date --iso-8601=seconds)] completed all variants"

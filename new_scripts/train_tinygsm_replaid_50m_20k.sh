#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 GPU_ID [RUN_LABEL]" >&2
  exit 2
fi

repo_root="/home/patrick/duo"
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
gpu_id=$1
run_label=${2:-replaid50m-lr3e-4-special1e-2-b64}
output_dir="${storage_root}/new_training_runs/tinygsm_replaid_50m/${run_label}"

mkdir -p "${output_dir}"
cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec .venv/bin/python main.py \
  +experiment=tinygsm_replaid_50m \
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
  wandb.project=tinygsm-replaid-50m \
  wandb.group=512-20k-step \
  wandb.name="${run_label}" \
  wandb.tags='[tinygsm,replaid,continuous-diffusion,50m,context512,20k-steps,backbone-lr-3e-4,special-lr-1e-2]'

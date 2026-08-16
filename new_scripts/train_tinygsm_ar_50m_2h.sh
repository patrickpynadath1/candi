#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 GPU_ID LEARNING_RATE RUN_LABEL [WANDB_RUN_ID]" >&2
  exit 2
fi

gpu_id=$1
learning_rate=$2
run_label=$3
output_dir="/home/patrick/duo/new_smoke_runs/tinygsm_ar_50m_2h/${run_label}"
checkpoint_path="${output_dir}/checkpoints/last.ckpt"
wandb_args=()
if [[ $# -eq 4 ]]; then
  wandb_args+=("wandb.id=$4" "+wandb.resume=allow")
fi

if [[ ! -f "${checkpoint_path}" ]]; then
  echo "missing resume checkpoint: ${checkpoint_path}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python main.py \
  +experiment=tinygsm_ar_50m \
  scratch_dir="${output_dir}" \
  checkpointing.save_dir="${output_dir}" \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path="${checkpoint_path}" \
  trainer.devices=1 \
  trainer.accumulate_grad_batches=1 \
  trainer.max_steps=150000 \
  trainer.val_check_interval=1000 \
  trainer.num_sanity_val_steps=0 \
  trainer.limit_val_batches=8 \
  loader.global_batch_size=64 \
  loader.eval_global_batch_size=64 \
  loader.num_workers=4 \
  data.validation_examples=2048 \
  optim.lr="${learning_rate}" \
  sampling.predictor=ar_cached \
  callbacks.gsm8k_eval.every_n_train_steps=2000 \
  callbacks.gsm8k_eval.subset_size=128 \
  callbacks.gsm8k_eval.batch_size=16 \
  callbacks.gsm8k_eval.num_steps=32 \
  callbacks.gsm8k_eval.log_examples=16 \
  callbacks.gsm8k_eval.full_eval_at_end=false \
  wandb.project=tinygsm-ar-50m-baselines \
  wandb.group=512-150k-step-sweep-temp-0.2 \
  wandb.name="${run_label}" \
  wandb.tags="[tinygsm,ar,50m,context512,150k-steps,temp-0.2,lr-${learning_rate}]" \
  "${wandb_args[@]}"

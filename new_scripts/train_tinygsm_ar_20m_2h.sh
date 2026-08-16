#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU_ID LEARNING_RATE RUN_LABEL" >&2
  exit 2
fi

gpu_id=$1
learning_rate=$2
run_label=$3
output_dir="/home/patrick/duo/new_smoke_runs/tinygsm_ar_20m_2h/${run_label}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python main.py \
  +experiment=tinygsm_ar_20m \
  scratch_dir="${output_dir}" \
  checkpointing.save_dir="${output_dir}" \
  checkpointing.resume_from_ckpt=false \
  trainer.devices=1 \
  trainer.accumulate_grad_batches=1 \
  trainer.max_steps=250000 \
  +trainer.max_time=00:02:00:00 \
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
  wandb.project=tinygsm-ar-20m-baselines \
  wandb.group=512-two-hour-sweep \
  wandb.name="${run_label}" \
  wandb.tags="[tinygsm,ar,20m,context512,two-hour,lr-${learning_rate}]"

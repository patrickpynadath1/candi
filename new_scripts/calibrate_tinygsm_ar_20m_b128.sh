#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5
export TOKENIZERS_PARALLELISM=false

uv run python main.py \
  +experiment=tinygsm_ar_20m \
  scratch_dir=/home/patrick/duo/new_smoke_runs/tinygsm_ar_20m_b128 \
  checkpointing.save_dir=/home/patrick/duo/new_smoke_runs/tinygsm_ar_20m_b128 \
  checkpointing.resume_from_ckpt=false \
  trainer.devices=1 \
  trainer.accumulate_grad_batches=1 \
  trainer.max_steps=10 \
  trainer.val_check_interval=10 \
  trainer.num_sanity_val_steps=0 \
  trainer.limit_val_batches=1 \
  loader.global_batch_size=128 \
  loader.eval_global_batch_size=64 \
  loader.num_workers=4 \
  data.validation_examples=64 \
  callbacks.gsm8k_eval.every_n_train_steps=1000 \
  callbacks.gsm8k_eval.full_eval_at_end=false \
  wandb.project=tinygsm-training-smokes \
  wandb.group=ar-20m-calibration \
  wandb.name=ar-20m-b128-calibration \
  wandb.tags='[tinygsm,ar,20m,calibration,b128]'

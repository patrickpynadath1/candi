#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python main.py \
  +experiment=tinygsm_ar_50m \
  scratch_dir=/home/patrick/duo/new_smoke_runs/tinygsm_ar_50m_calibration \
  checkpointing.save_dir=/home/patrick/duo/new_smoke_runs/tinygsm_ar_50m_calibration \
  checkpointing.resume_from_ckpt=false \
  trainer.devices=1 \
  trainer.accumulate_grad_batches=1 \
  trainer.max_steps=5 \
  trainer.val_check_interval=5 \
  trainer.num_sanity_val_steps=0 \
  trainer.limit_val_batches=2 \
  loader.global_batch_size=64 \
  loader.eval_global_batch_size=64 \
  loader.num_workers=4 \
  data.validation_examples=64 \
  callbacks.gsm8k_eval.every_n_train_steps=1000 \
  callbacks.gsm8k_eval.full_eval_at_end=false \
  +wandb.mode=disabled

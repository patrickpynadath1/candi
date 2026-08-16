#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5
export TOKENIZERS_PARALLELISM=false

uv run python main.py \
  +experiment=tinygsm_ar \
  scratch_dir=/home/patrick/duo/new_smoke_runs/tinygsm_ar_callback \
  checkpointing.save_dir=/home/patrick/duo/new_smoke_runs/tinygsm_ar_callback \
  checkpointing.resume_from_ckpt=false \
  trainer.devices=1 \
  trainer.accumulate_grad_batches=1 \
  trainer.max_steps=4 \
  trainer.val_check_interval=2 \
  trainer.num_sanity_val_steps=0 \
  loader.global_batch_size=2 \
  loader.eval_global_batch_size=2 \
  loader.num_workers=0 \
  +data.max_examples=32 \
  data.val_ratio=0.25 \
  sampling.predictor=ar_cached \
  callbacks.gsm8k_eval.every_n_train_steps=2 \
  callbacks.gsm8k_eval.subset_size=4 \
  callbacks.gsm8k_eval.batch_size=2 \
  callbacks.gsm8k_eval.num_steps=2 \
  callbacks.gsm8k_eval.log_examples=4 \
  callbacks.gsm8k_eval.full_eval_at_end=false \
  wandb.project=tinygsm-training-smokes \
  wandb.group=ar-callback \
  wandb.name=ar-512-callback-smoke \
  wandb.tags='[tinygsm,ar,callback,smoke]'

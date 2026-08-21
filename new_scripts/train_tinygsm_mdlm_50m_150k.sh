#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 GPU_ID [RUN_LABEL]" >&2
  exit 2
fi

repo_root="/home/patrick/duo"
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
gpu_id=$1
run_label=${2:-mdlm50m-lr3e-4-b64}
output_dir="${storage_root}/new_training_runs/tinygsm_mdlm_50m/${run_label}"

mkdir -p "${output_dir}"
cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec .venv/bin/python main.py \
  +experiment=tinygsm_mdlm_50m \
  scratch_dir="${output_dir}" \
  checkpointing.save_dir="${output_dir}" \
  checkpointing.resume_from_ckpt=false \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=2000 \
  trainer.devices=1 \
  trainer.accumulate_grad_batches=1 \
  trainer.max_steps=20000 \
  optim.lr=3e-4 \
  data.train_on_pad=false \
  algo.mask_after_added_tokens=true \
  callbacks.gsm8k_eval.every_n_train_steps=2000 \
  callbacks.gsm8k_eval.subset_size=128 \
  callbacks.gsm8k_eval.batch_size=16 \
  callbacks.gsm8k_eval.num_steps=32 \
  callbacks.gsm8k_eval.log_examples=16 \
  callbacks.gsm8k_eval.full_eval_at_end=false \
  wandb.project=tinygsm-mdlm-50m-baselines \
  sampling.predictor=entropy_bounded \
  sampling.entropy_bound=0.1 \
  wandb.group=512-20k-step-eb-0.1 \
  wandb.name="${run_label}" \
  wandb.tags='[tinygsm,mdlm,masked-diffusion,entropy-bounded,gamma-0.1,50m,context512,20k-steps,lr-3e-4]'

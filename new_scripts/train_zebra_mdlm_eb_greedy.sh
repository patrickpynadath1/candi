#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_dir="${RUN_DIR:-$PWD/runs/zebra_mdlm_eb_greedy_20k_seed1}"
gpu="${GPU:-0}"
mkdir -p "$run_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online
export WANDB_RESUME=allow
export PYTHONUNBUFFERED=1

exec .venv/bin/python main.py \
  +experiment=zebra_mdlm_eb_greedy \
  trainer.devices=1 \
  trainer.num_nodes=1 \
  checkpointing.resume_from_ckpt=false \
  checkpointing.save_dir="$run_dir" \
  hydra.run.dir="$run_dir" \
  wandb.project=reasoning-masked-diffusion-ablations \
  +wandb.entity=rz-text-dlp \
  wandb.id=zebra-mixed-mdlm-eb-greedy-gamma0p1-seed1 \
  wandb.name=zebra_mixed_mdlm_eb_greedy_gamma0p1_seed1 \
  wandb.group=zebra_mixed_masked_diffusion \
  wandb.job_type=train \
  wandb.tags='[zebra_mixed,mdlm,entropy_bounded,greedy_argmax,packed_v2,decontaminated]'

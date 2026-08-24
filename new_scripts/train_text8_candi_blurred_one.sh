#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID INPUT_DIM [--smoke] [--temperature T]" >&2
  exit 2
fi
gpu_id=$1
input_dim=$2
shift 2
smoke=false
temperature=1.0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) smoke=true; shift ;;
    --temperature) temperature=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
label="candi_blurred_d${input_dim}_text8_5k"
extra=()
if ${smoke}; then
  label="smoke_${label}"
  extra=(
    trainer.max_steps=2
    trainer.val_check_interval=2
    loader.global_batch_size=8
    loader.eval_global_batch_size=8
    loader.num_workers=0
    callbacks.text8_sampling.every_n_train_steps=2
    callbacks.text8_sampling.num_samples=4
    callbacks.text8_sampling.batch_size=2
    callbacks.text8_sampling.num_steps=8
    callbacks.text8_sampling.samplers.1.entropy_bound=100
    callbacks.checkpoint_every_n_steps.every_n_train_steps=2
    wandb=null
  )
fi
run_dir="${storage_root}/new_runs/${label}"
mkdir -p "${run_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config_text8_frontier \
  +experiment=text8_candi_blurred_frontier \
  "algo.input_embedding_dim=${input_dim}" \
  "callbacks.text8_sampling.samplers.0.temperature=${temperature}" \
  "checkpointing.save_dir=${run_dir}" \
  checkpointing.resume_from_ckpt=true \
  "wandb.name=${label}" \
  "wandb.tags=[text8,tiny-dit,5k,candi-blurred,din-${input_dim}]" \
  "hydra.run.dir=${run_dir}" \
  "${extra[@]}"

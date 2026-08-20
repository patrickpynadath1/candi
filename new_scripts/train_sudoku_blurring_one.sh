#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID LABEL [D_IN] [dct|isotropic] [live|detached] [HYDRA_OVERRIDE ...]" >&2
  exit 2
fi

gpu_id=$1
label=$2
input_dim=${3:-4}
basis=${4:-dct}
embedding_mode=${5:-live}
if [[ $# -ge 5 ]]; then
  shift 5
else
  shift "$#"
fi
extra_overrides=("$@")
if [[ "${basis}" != "dct" && "${basis}" != "isotropic" ]]; then
  echo "basis must be dct or isotropic" >&2
  exit 2
fi
if [[ "${embedding_mode}" != "live" && "${embedding_mode}" != "detached" ]]; then
  echo "embedding mode must be live or detached" >&2
  exit 2
fi

detach=false
if [[ "${embedding_mode}" == "detached" ]]; then
  detach=true
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
run_dir="${storage_root}/new_sudoku_runs/blurring/${label}"
dataset_cache="${storage_root}/data_cache"
mkdir -p "${run_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config \
  +experiment=sudoku_blurring_hard \
  "data.cache_dir=${dataset_cache}" \
  "algo.input_embedding_dim=${input_dim}" \
  "algo.basis=${basis}" \
  "algo.detach_forward_embedding=${detach}" \
  wandb.project=sudoku-blurring-hard-20k \
  "wandb.group=blurring-${basis}" \
  "wandb.name=${label}" \
  "wandb.tags=[sudoku,hard,blurring,${basis},din-${input_dim},${embedding_mode},20k-steps]" \
  checkpointing.resume_from_ckpt=false \
  "checkpointing.save_dir=${run_dir}" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  +trainer.check_val_every_n_epoch=null \
  "hydra.run.dir=${run_dir}" \
  "${extra_overrides[@]}"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_ID ALGORITHM" >&2
  exit 2
fi

gpu_id=$1
algorithm=$2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_dir="${repo_root}/new_runs/${algorithm}"
mkdir -p "${run_dir}"

extra=()
if [[ "${algorithm}" == "candi" ]]; then
  extra+=(algo.max_percentile=0.4 algo.min_percentile=0.05)
elif [[ "${algorithm}" == "flm" ]]; then
  extra+=(model=tiny_flm model.length=128)
elif [[ "${algorithm}" == "sfm" ]]; then
  extra+=(model=tiny_sphere_dit model.length=128)
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config_text8_frontier \
  "algo=${algorithm}" \
  "checkpointing.save_dir=${run_dir}" \
  "hydra.run.dir=${run_dir}" \
  "wandb.name=${algorithm}-tiny-dit-text8-5k" \
  "${extra[@]}"

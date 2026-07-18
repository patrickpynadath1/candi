#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU_ID ALGORITHM NFE TEMPERATURE" >&2
  exit 2
fi

gpu_id=$1
algorithm=$2
nfe=$3
temperature=$4
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config_algorithm=${algorithm}
if [[ "${algorithm}" == "cont_diff_embed" ]]; then
  config_algorithm=embedding_diffusion
fi
checkpoint="${repo_root}/new_runs/${config_algorithm}/checkpoints/last.ckpt"
output_dir="${repo_root}/new_results/${algorithm}"
output="${output_dir}/nfe_${nfe}_temp_${temperature}.json"
mkdir -p "${output_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" uv run python "${repo_root}/main.py" \
  --config-name=config_text8_frontier \
  mode=sample_eval \
  "algo=${config_algorithm}" \
  "algo.temp=${temperature}" \
  "sampling.steps=${nfe}" \
  sampling.num_sample_batches=1 \
  "eval.checkpoint_path=${checkpoint}" \
  "eval.generated_samples_path=${output}" \
  "hydra.run.dir=${repo_root}/new_eval_runs/${algorithm}/nfe_${nfe}_temp_${temperature}" \
  wandb=null

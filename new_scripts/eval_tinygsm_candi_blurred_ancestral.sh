#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 GPU_ID [MAX_EXAMPLES] [STEPS]" >&2
  exit 2
fi

gpu_id=$1
max_examples=${2:-64}
steps=${3:-32}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=/home/patrick/.cache/discrete_diffusion/openwebtext-train/candi-blurred/checkpoints/last.ckpt
label="candi_blurred_ancestral_temp_0p1_n${max_examples}_steps${steps}"
output_dir="${repo_root}/new_eval_results/tinygsm_candi_blurred/${label}"
run_dir="${repo_root}/new_eval_runs/tinygsm_candi_blurred/${label}"

[[ -f "${checkpoint}" ]] || {
  echo "missing checkpoint: ${checkpoint}" >&2
  exit 1
}

# CANDI's joint continuous/discrete ancestral kernel is registered as candi.
CUDA_VISIBLE_DEVICES="${gpu_id}" \
HF_DATASETS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
"${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config \
  mode=gsm8k_eval \
  data=gsm8k-test \
  data.cache_dir="${repo_root}/data_cache" \
  model=small \
  model.length=512 \
  algo=candi_blurred \
  algo.temp=0.1 \
  algo.input_embedding_dim=512 \
  algo.output_embedding_dim=512 \
  algo.blurring.terminal_mean=0.005 \
  algo.blurring.noise_power=1.0 \
  sampling.predictor=candi \
  sampling.trace_argmax_changes=true \
  "sampling.steps=${steps}" \
  loader.eval_global_batch_size=64 \
  loader.num_workers=1 \
  "eval.checkpoint_path=${checkpoint}" \
  "gsm8k.max_examples=${max_examples}" \
  gsm8k.subset_seed=42 \
  gsm8k.scoring=code \
  "gsm8k.output_dir=${output_dir}" \
  "hydra.run.dir=${run_dir}" \
  wandb=null

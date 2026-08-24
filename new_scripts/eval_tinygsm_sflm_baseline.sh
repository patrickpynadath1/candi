#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "usage: $0 GPU_ID {mdlm|duo} [MAX_EXAMPLES] [STEPS] [TEMPERATURE]" >&2
  exit 2
fi

gpu_id=$1
model_name=$2
max_examples=${3:-64}
steps=${4:-32}
temperature=${5:-0.1}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint_root=/home/patrick/.cache/discrete_diffusion/pretrained/s-flm/tinygsm

case "${model_name}" in
  mdlm)
    algorithm=mdlm
    checkpoint="${checkpoint_root}/mdlm.ckpt"
    ;;
  duo)
    algorithm=duo_base
    checkpoint="${checkpoint_root}/duo.ckpt"
    ;;
  *)
    echo "unknown model: ${model_name}; expected mdlm or duo" >&2
    exit 2
    ;;
esac

[[ -f "${checkpoint}" ]] || {
  echo "missing checkpoint: ${checkpoint}" >&2
  exit 1
}

temperature_label=${temperature//./p}
label="${model_name}_ancestral_temp_${temperature_label}_n${max_examples}_steps${steps}"
output_dir="${repo_root}/new_eval_results/tinygsm_sflm/${label}"
run_dir="${repo_root}/new_eval_runs/tinygsm_sflm/${label}"

CUDA_VISIBLE_DEVICES="${gpu_id}" \
HF_DATASETS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
"${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config \
  mode=gsm8k_eval \
  data=gsm8k-test-sflm \
  data.cache_dir="${repo_root}/data_cache" \
  model=small \
  model.length=512 \
  +model.sflm_compat=true \
  "algo=${algorithm}" \
  "algo.temp=${temperature}" \
  noise.eps=0 \
  sampling.predictor=ancestral \
  sampling.noise_removal=ancestral \
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

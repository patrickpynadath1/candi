#!/usr/bin/env bash
set -euo pipefail

gpu_id=${1:-2}
max_examples=${2:-64}
method=${3:-eb}
temperature=${4:-0.1}
gamma=${5:-0.1}
gamma_label=${gamma//./p}
temp_label=${temperature//./p}
case "${method}" in
  eb) predictor=candi_embedding_entropy_bounded; steps=512
      label="eb_categorical_gamma${gamma_label}_temp${temp_label}" ;;
  ancestral) predictor=candi_embedding; steps=128
      label="ancestral_nfe128_temp${temp_label}" ;;
  *) echo "method must be eb or ancestral" >&2; exit 2 ;;
esac
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=/home/patrick/.cache/discrete_diffusion/openwebtext-train/candi_geo_rank_tinygsm/checkpoints/last.ckpt
run_dir="${repo_root}/runs/candi_geo_rank_tinygsm/${label}_n${max_examples}"
mkdir -p "${run_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
"${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config mode=gsm8k_eval \
  data=gsm8k-test data.cache_dir="${repo_root}/data_cache" \
  model=small model.length=512 \
  algo=candi_embedding algo.interpolation_schedule=geometric_rank \
  +algo.legacy_architecture=false algo.temp="${temperature}" noise.eps=0.0 \
  sampling.predictor="${predictor}" \
  sampling.entropy_bound="${gamma}" sampling.greedy_clean_estimate=false \
  sampling.steps="${steps}" \
  trainer.devices=1 trainer.accumulate_grad_batches=1 \
  loader.eval_global_batch_size=8 loader.num_workers=0 \
  eval.checkpoint_path="${checkpoint}" \
  gsm8k.max_examples="${max_examples}" gsm8k.subset_seed=42 gsm8k.scoring=code \
  gsm8k.output_dir="${run_dir}" hydra.run.dir="${run_dir}" wandb=null \
  2>&1 | tee "${run_dir}/inference.log"

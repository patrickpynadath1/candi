#!/usr/bin/env bash
set -euo pipefail

gpu_id=${1:-2}
max_examples=${2:-64}
gamma=${3:-0.1}
gamma_label=${gamma//./p}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=/home/patrick/.cache/discrete_diffusion/openwebtext-train/candi_geo_rank_tinygsm/checkpoints/last.ckpt
run_dir="${repo_root}/runs/candi_geo_rank_tinygsm/eb_greedy_gamma${gamma_label}_n${max_examples}"
mkdir -p "${run_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
"${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config mode=gsm8k_eval \
  data=gsm8k-test data.cache_dir="${repo_root}/data_cache" \
  model=small model.length=512 \
  algo=candi_embedding algo.interpolation_schedule=geometric_rank \
  +algo.legacy_architecture=false algo.temp=1.0 noise.eps=0.0 \
  sampling.predictor=candi_embedding_entropy_bounded \
  sampling.entropy_bound="${gamma}" sampling.greedy_clean_estimate=true \
  sampling.steps=512 \
  trainer.devices=1 trainer.accumulate_grad_batches=1 \
  loader.eval_global_batch_size=8 loader.num_workers=0 \
  eval.checkpoint_path="${checkpoint}" \
  gsm8k.max_examples="${max_examples}" gsm8k.subset_seed=42 gsm8k.scoring=code \
  gsm8k.output_dir="${run_dir}" hydra.run.dir="${run_dir}" wandb=null \
  2>&1 | tee "${run_dir}/inference.log"

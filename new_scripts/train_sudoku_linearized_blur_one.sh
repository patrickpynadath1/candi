#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID pure|candi [--smoke] [--input-dim N] [--rho R] [--schedule geometric|linearized]" >&2
  exit 2
fi
gpu_id=$1
variant=$2
shift 2
smoke=false
input_dim=""
rho=1.0
schedule=linearized
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) smoke=true; shift ;;
    --input-dim) input_dim=$2; shift 2 ;;
    --rho) rho=$2; shift 2 ;;
    --schedule) schedule=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ "${variant}" != "pure" && "${variant}" != "candi" ]]; then
  echo "variant must be pure or candi" >&2
  exit 2
fi
if [[ -z "${input_dim}" ]]; then
  input_dim=512
fi
case "${rho}" in
  0|0.0|0.25|0.5|0.75|1|1.0) ;;
  *) echo "rho must be one of 0, 0.25, 0.5, 0.75, 1" >&2; exit 2 ;;
esac
if [[ "${schedule}" != "geometric" && "${schedule}" != "linearized" ]]; then
  echo "schedule must be geometric or linearized" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
project=sudoku-blurring-hard-20k
common=(
  "data.cache_dir=${storage_root}/data_cache"
  "algo.input_embedding_dim=${input_dim}"
  trainer.max_steps=20000
  trainer.val_check_interval=20000
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000
  +trainer.check_val_every_n_epoch=null
  checkpointing.resume_from_ckpt=false
  "wandb.project=${project}"
)

if [[ "${variant}" == "pure" ]]; then
  experiment=sudoku_blurring_hard
  label="blurring_${schedule}_rho${rho}_d${input_dim}_hard"
  common+=(
    "algo.feature_rho=${rho}"
    algo.detach_forward_embedding=false
    algo.linearized_crossing_min=0.05
    algo.linearized_crossing_max=0.95
    sampling.predictor=blurring_ode
    sampling.greedy_clean_estimate=false
    sampling.omit_reblur=false
    sampling.steps=64
    callbacks.sudoku_eval.num_steps=64
    wandb.group=two-axis-blur-rho
    "wandb.tags=[sudoku,hard,blurring,pure-continuous,${schedule},rho-${rho},reblur,din-${input_dim},live]"
  )
else
  experiment=sudoku_candi_blurred_hard_eb
  label="candi_blurred_${schedule}_rho${rho}_d${input_dim}_hard_eb"
  common+=(
    "algo.blurring.feature_rho=${rho}"
    algo.blurring.detach_forward_embedding=false
    algo.blurring.linearized_crossing_min=0.05
    algo.blurring.linearized_crossing_max=0.95
    sampling.predictor=candi_entropy_bounded
    sampling.entropy_bound=0.1
    sampling.greedy_clean_estimate=true
    sampling.steps=180
    callbacks.sudoku_eval.num_steps=180
    wandb.group=two-axis-blur-rho
    "wandb.tags=[sudoku,hard,candi-blurred,${schedule},rho-${rho},entropy-bounded,argmax,reblur,din-${input_dim},live]"
  )
fi

if [[ "${schedule}" == "linearized" ]]; then
  if [[ "${variant}" == "pure" ]]; then
    common+=(algo.time_warp=linearized_crossings algo.crossing_max_decile_fraction=0.15)
  else
    common+=(algo.blurring.time_warp=linearized_crossings algo.blurring.crossing_max_decile_fraction=0.15)
  fi
else
  if [[ "${variant}" == "pure" ]]; then
    common+=(algo.time_warp=none algo.crossing_max_decile_fraction=0.45)
  else
    common+=(algo.blurring.time_warp=none algo.blurring.crossing_max_decile_fraction=0.45)
  fi
fi

if ${smoke}; then
  label="smoke_${label}"
  common+=(
    trainer.max_steps=2
    trainer.val_check_interval=2
    data.num_train=64
    data.num_valid=32
    loader.global_batch_size=8
    loader.eval_global_batch_size=8
    loader.num_workers=0
    callbacks.sudoku_eval.every_n_train_steps=2
    callbacks.sudoku_eval.subset_size=8
    callbacks.sudoku_eval.batch_size=4
    callbacks.checkpoint_every_n_steps.every_n_train_steps=2
    wandb=null
  )
  if [[ "${variant}" == "pure" ]]; then
    common+=(sampling.steps=4 callbacks.sudoku_eval.num_steps=4)
  fi
fi

run_dir="${storage_root}/new_sudoku_runs/${label}"
mkdir -p "${run_dir}"
CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config \
  "+experiment=${experiment}" \
  "checkpointing.save_dir=${run_dir}" \
  "wandb.name=${label}" \
  "hydra.run.dir=${run_dir}" \
  "${common[@]}"

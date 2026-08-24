#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/train_text8_rolling_diff.sh [AR_GPU] [BLOCK_AR_GPU] [ANNEALED_GPU]
# Example:
#   ./scripts/train_text8_rolling_diff.sh 3 5 7

AR_GPU="${1:-3}"
BLOCK_AR_GPU="${2:-5}"
ANNEALED_GPU="${3:-7}"
RUN_ROOT="${RUN_ROOT:-${PWD}/new_runs/rolling-diff}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  # The environment must be synchronized once before launching. --no-sync
  # prevents three concurrent jobs from racing to mutate the shared .venv.
  PYTHON_CMD=(uv run --no-sync python)
fi

COMMON_OVERRIDES=(
  mode=train
  data=text8
  model=tiny
  seed=17
  batch_size=128
  loader.global_batch_size=128
  loader.eval_global_batch_size=128
  loader.num_workers=4
  model.length=128
  model.hidden_size=128
  model.cond_dim=128
  model.n_blocks=4
  model.n_heads=4
  model.dropout=0.0
  +model.normalize_embeddings=true
  trainer.devices=1
  trainer.num_nodes=1
  trainer.accumulate_grad_batches=1
  trainer.max_steps=10000
  trainer.val_check_interval=1000
  trainer.check_val_every_n_epoch=null
  trainer.num_sanity_val_steps=0
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000
  eval.generate_samples=false
  wandb.project=rolling-diff
  wandb.group=text8-10k-equivalence-and-annealing
)

run_job() {
  local gpu="$1"
  local run_name="$2"
  shift 2

  local run_dir="${RUN_ROOT}/${run_name}"
  mkdir -p "${run_dir}"
  echo "Starting ${run_name} on GPU ${gpu}; outputs: ${run_dir}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_CMD[@]}" main.py \
    --config-name=config_text8_frontier \
    "${COMMON_OVERRIDES[@]}" \
    "wandb.name=${run_name}" \
    "wandb.id=${run_name}-seed17" \
    "hydra.run.dir=${run_dir}" \
    "checkpointing.save_dir=${run_dir}" \
    "checkpointing.resume_ckpt_path=${run_dir}/checkpoints/last.ckpt" \
    "$@" \
    >"${run_dir}/train.log" 2>&1 &

  LAST_PID=$!
}

PIDS=()
NAMES=()

# Ordinary AR control. Normalized embeddings are shared by all three arms so
# that the zero-noise block run differs only in sequence layout and its mask.
run_job "${AR_GPU}" text8-ar-10k \
  algo=ar \
  sampling.predictor=ar \
  'wandb.tags=[text8,rolling-diff,ar,10k]'
PIDS+=("${LAST_PID}")
NAMES+=(text8-ar-10k)

# Doubled block model with corruption permanently disabled. This should track
# the AR objective, modulo attention-kernel floating-point differences.
run_job "${BLOCK_AR_GPU}" text8-block-zero-noise-10k \
  algo=block_diffusion \
  sampling.predictor=block_diffusion \
  algo.block_size=8 \
  algo.max_noise_level=0.0 \
  algo.noise_curriculum_steps=0 \
  'wandb.tags=[text8,rolling-diff,block-diffusion,zero-noise,10k]'
PIDS+=("${LAST_PID}")
NAMES+=(text8-block-zero-noise-10k)

# Real rolling block-diffusion training. During the first 2,500 optimizer steps,
# block maxima are sampled uniformly from [0, min(s / 2500, 1)]. For the
# remaining 7,500 steps, maxima use the full [0, 1] range. Within each block,
# sorted-uniform position times form a non-decreasing schedule ending at the
# sampled maximum.
run_job "${ANNEALED_GPU}" text8-block-annealed-10k \
  algo=block_diffusion \
  sampling.predictor=block_diffusion \
  algo.block_size=8 \
  algo.max_noise_level=1.0 \
  algo.noise_curriculum_steps=2500 \
  'wandb.tags=[text8,rolling-diff,block-diffusion,annealed,10k]'
PIDS+=("${LAST_PID}")
NAMES+=(text8-block-annealed-10k)

status=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "Finished ${NAMES[$index]}"
  else
    echo "FAILED ${NAMES[$index]} (see ${RUN_ROOT}/${NAMES[$index]}/train.log)" >&2
    status=1
  fi
done

exit "${status}"

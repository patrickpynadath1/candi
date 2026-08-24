#!/usr/bin/env bash
set -euo pipefail

# Four matched block-size-16 runs with different clean AR loss weights.
# Usage: ./scripts/train_text8_block_loss_weight_ablation.sh [GPU0 GPU1 GPU2 GPU3]

GPUS=("${1:-3}" "${2:-4}" "${3:-5}" "${4:-7}")
WEIGHTS=("0.0" "0.1" "0.25" "0.5")
LABELS=("0" "010" "025" "050")
RUN_ROOT="${RUN_ROOT:-${PWD}/new_runs/rolling-diff-block16-anchored-adaln-ablation}"
PYTHON_CMD=(uv run --no-sync python)

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
  algo=block_diffusion
  sampling.predictor=block_diffusion
  algo.block_size=16
  algo.max_noise_level=1.0
  algo.noise_curriculum_steps=2500
  algo.gradient_cosine_every=100
  wandb.project=rolling-diff
  wandb.group=text8-block16-anchored-adaln-clean-loss-ablation
)

PIDS=()
NAMES=()
for index in "${!WEIGHTS[@]}"; do
  weight="${WEIGHTS[$index]}"
  run_name="text8-block16-adaln-clean${LABELS[$index]}-10k"
  run_dir="${RUN_ROOT}/${run_name}"
  mkdir -p "${run_dir}"
  echo "Starting ${run_name} on GPU ${GPUS[$index]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$index]}" "${PYTHON_CMD[@]}" main.py \
    --config-name=config_text8_frontier \
    "${COMMON_OVERRIDES[@]}" \
    "algo.clean_loss_weight=${weight}" \
    "wandb.name=${run_name}" \
    "wandb.id=${run_name}-seed17" \
    'wandb.tags=[text8,rolling-diff,block16,anchored-tokenwise-adaln,loss-weight-ablation,10k]' \
    "hydra.run.dir=${run_dir}" \
    "checkpointing.save_dir=${run_dir}" \
    "checkpointing.resume_ckpt_path=${run_dir}/checkpoints/last.ckpt" \
    >"${run_dir}/train.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${run_name}")
done

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

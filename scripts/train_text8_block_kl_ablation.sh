#!/usr/bin/env bash
set -euo pipefail

# Factorized KL ablation for the best block-16 clean-loss setting.
# Usage: ./scripts/train_text8_block_kl_ablation.sh [GPU0 GPU1 GPU2 GPU3 GPU4 GPU5]

GPUS=("${1:-2}" "${2:-3}" "${3:-4}" "${4:-5}" "${5:-6}" "${6:-7}")
RUN_ROOT="${RUN_ROOT:-${PWD}/new_runs/rolling-diff-block16-kl-ablation}"
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
  algo.clean_loss_weight=0.25
  algo.gradient_cosine_every=100
  algo.logit_kl_position_weighting=noise
  wandb.project=rolling-diff
  wandb.group=text8-block16-logit-kl-ablation
)

run_wave() {
  local directions=("$1" "$4" "$7" "${10}" "${13}" "${16}")
  local weights=("$2" "$5" "$8" "${11}" "${14}" "${17}")
  local temperatures=("$3" "$6" "$9" "${12}" "${15}" "${18}")
  local pids=()
  local names=()

  for index in "${!GPUS[@]}"; do
    direction="${directions[$index]}"
    weight="${weights[$index]}"
    temperature="${temperatures[$index]}"
    weight_label="${weight//./p}"
    temperature_label="${temperature//./p}"
    run_name="text8-block16-kl-${direction}-w${weight_label}-t${temperature_label}-10k"
    run_dir="${RUN_ROOT}/${run_name}"
    mkdir -p "${run_dir}"
    echo "Starting ${run_name} on GPU ${GPUS[$index]}"
    CUDA_VISIBLE_DEVICES="${GPUS[$index]}" "${PYTHON_CMD[@]}" main.py \
      --config-name=config_text8_frontier \
      "${COMMON_OVERRIDES[@]}" \
      "algo.logit_kl_direction=${direction}" \
      "algo.logit_kl_weight=${weight}" \
      "algo.logit_kl_temperature=${temperature}" \
      "wandb.name=${run_name}" \
      "wandb.id=${run_name}-seed17" \
      'wandb.tags=[text8,rolling-diff,block16,anchored-tokenwise-adaln,logit-kl,10k]' \
      "hydra.run.dir=${run_dir}" \
      "checkpointing.save_dir=${run_dir}" \
      "checkpointing.resume_ckpt_path=${run_dir}/checkpoints/last.ckpt" \
      >"${run_dir}/train.log" 2>&1 &
    pids+=("$!")
    names+=("${run_name}")
  done

  local status=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "Finished ${names[$index]}"
    else
      echo "FAILED ${names[$index]} (see ${RUN_ROOT}/${names[$index]}/train.log)" >&2
      status=1
    fi
  done
  return "${status}"
}

# Direction/weight sweep at the native distribution temperature.
run_wave \
  forward 0.05 1.0 \
  forward 0.1 1.0 \
  forward 0.25 1.0 \
  reverse 0.05 1.0 \
  reverse 0.1 1.0 \
  reverse 0.25 1.0

# Temperature sweep at the central KL coefficient.
run_wave \
  forward 0.1 0.5 \
  forward 0.1 2.0 \
  forward 0.1 4.0 \
  reverse 0.1 0.5 \
  reverse 0.1 2.0 \
  reverse 0.1 4.0

#!/usr/bin/env bash
set -euo pipefail

# Compare delayed block-diffusion noise curricula with KL disabled.
# Usage: ./scripts/train_text8_block_delayed_noise_ablation.sh [GPU0 GPU1 GPU2]

GPUS=("${1:-3}" "${2:-4}" "${3:-5}")
STARTS=(2500 5000 2500)
ENDS=(5000 7500 5000)
NOISE_MODES=(monotone monotone sequence_constant)
RUN_ROOT="${RUN_ROOT:-${PWD}/new_runs/rolling-diff-block16-delayed-noise-rngsafe-ablation}"
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
  +callbacks.rolling_generation._target_=tasks.text8_callback.Text8SamplingCallback
  +callbacks.rolling_generation.every_n_train_steps=1000
  +callbacks.rolling_generation.num_samples=32
  +callbacks.rolling_generation.batch_size=32
  +callbacks.rolling_generation.num_steps=128
  +callbacks.rolling_generation.log_examples=4
  +callbacks.rolling_generation.at_validation_end=true
  +callbacks.rolling_generation.seed=23
  +callbacks.rolling_generation.samplers=null
  eval.generate_samples=false
  algo=block_diffusion
  sampling.predictor=block_diffusion
  algo.block_size=16
  algo.max_noise_level=1.0
  algo.clean_loss_weight=0.25
  algo.logit_kl_weight=0.0
  algo.gradient_cosine_every=100
  algo.temp=0.1
  algo.rolling_warmup_steps=8
  algo.rolling_refinement_steps=1
  algo.rolling_x0_projection=argmax
  wandb.project=rolling-diff
  wandb.group=text8-block16-delayed-noise-rngsafe-ablation
)

pids=()
names=()
for index in "${!GPUS[@]}"; do
  start="${STARTS[$index]}"
  end="${ENDS[$index]}"
  noise_mode="${NOISE_MODES[$index]}"
  run_name="text8-block16-rngsafe-noise${start}-${end}-${noise_mode}-10k"
  run_dir="${RUN_ROOT}/${run_name}"
  mkdir -p "${run_dir}"
  echo "Starting ${run_name} on GPU ${GPUS[$index]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$index]}" "${PYTHON_CMD[@]}" main.py \
    --config-name=config_text8_frontier \
    "${COMMON_OVERRIDES[@]}" \
    "algo.noise_curriculum_start_step=${start}" \
    "algo.noise_curriculum_end_step=${end}" \
    "algo.noise_level_mode=${noise_mode}" \
    "+callbacks.rolling_generation.output_dir=${run_dir}/generation_eval" \
    "wandb.name=${run_name}" \
    "wandb.id=${run_name}-seed17" \
    'wandb.tags=[text8,rolling-diff,block16,delayed-noise,no-kl,generation-during-training,10k]' \
    "hydra.run.dir=${run_dir}" \
    "checkpointing.save_dir=${run_dir}" \
    "checkpointing.resume_ckpt_path=${run_dir}/checkpoints/last.ckpt" \
    >"${run_dir}/train.log" 2>&1 &
  pids+=("$!")
  names+=("${run_name}")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Finished ${names[$index]}"
  else
    echo "FAILED ${names[$index]} (see ${RUN_ROOT}/${names[$index]}/train.log)" >&2
    status=1
  fi
done
exit "${status}"

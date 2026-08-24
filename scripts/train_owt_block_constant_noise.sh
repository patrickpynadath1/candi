#!/usr/bin/env bash
set -euo pipefail

# Train rolling block diffusion on OpenWebText with one shared noise level per
# sequence. All major scale/curriculum settings can be overridden via env vars.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 MAX_STEPS=1000000 \
#     NOISE_START_STEP=250000 NOISE_END_STEP=500000 \
#     ./scripts/train_owt_block_constant_noise.sh

MAX_STEPS="${MAX_STEPS:-1000000}"
NOISE_START_STEP="${NOISE_START_STEP:-250000}"
NOISE_END_STEP="${NOISE_END_STEP:-500000}"
VAL_EVERY="${VAL_EVERY:-10000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
DEVICES="${DEVICES:-1}"
NUM_NODES="${NUM_NODES:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-16}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
CLEAN_LOSS_WEIGHT="${CLEAN_LOSS_WEIGHT:-0.25}"
GENERATION_SAMPLES="${GENERATION_SAMPLES:-4}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-4}"
GENERATION_SEED="${GENERATION_SEED:-23}"
SEED="${SEED:-17}"
WANDB_PROJECT="${WANDB_PROJECT:-rolling-diff}"
WANDB_GROUP="${WANDB_GROUP:-owt-block-constant-noise}"
RUN_NAME="${RUN_NAME:-owt-block${BLOCK_SIZE}-constant-noise-${MAX_STEPS}}"
RUN_ROOT="${RUN_ROOT:-${PWD}/new_runs/rolling-diff-owt}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_NAME}}"

if (( NOISE_START_STEP < 0 || NOISE_END_STEP < NOISE_START_STEP )); then
  echo "Expected 0 <= NOISE_START_STEP <= NOISE_END_STEP" >&2
  exit 2
fi
if (( NOISE_END_STEP > MAX_STEPS )); then
  echo "NOISE_END_STEP must not exceed MAX_STEPS" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"

uv run --no-sync python main.py \
  --config-name=config \
  mode=train \
  data=openwebtext-split \
  model=small \
  algo=block_diffusion \
  sampling.predictor=block_diffusion \
  seed="${SEED}" \
  model.length="${SEQUENCE_LENGTH}" \
  +model.normalize_embeddings=true \
  loader.batch_size="${MICRO_BATCH_SIZE}" \
  loader.eval_batch_size="${EVAL_BATCH_SIZE}" \
  trainer.devices="${DEVICES}" \
  trainer.num_nodes="${NUM_NODES}" \
  trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_check_interval="${VAL_EVERY}" \
  trainer.check_val_every_n_epoch=null \
  trainer.num_sanity_val_steps=0 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps="${CHECKPOINT_EVERY}" \
  +callbacks.rolling_generation._target_=tasks.text8_callback.Text8SamplingCallback \
  +callbacks.rolling_generation.every_n_train_steps="${VAL_EVERY}" \
  +callbacks.rolling_generation.num_samples="${GENERATION_SAMPLES}" \
  +callbacks.rolling_generation.batch_size="${GENERATION_BATCH_SIZE}" \
  +callbacks.rolling_generation.num_steps="${SEQUENCE_LENGTH}" \
  +callbacks.rolling_generation.log_examples=4 \
  +callbacks.rolling_generation.at_validation_end=true \
  +callbacks.rolling_generation.seed="${GENERATION_SEED}" \
  +callbacks.rolling_generation.samplers=null \
  +callbacks.rolling_generation.output_dir="${RUN_DIR}/generation_eval" \
  eval.generate_samples=false \
  algo.block_size="${BLOCK_SIZE}" \
  algo.noise_level_mode=sequence_constant \
  algo.noise_curriculum_start_step="${NOISE_START_STEP}" \
  algo.noise_curriculum_end_step="${NOISE_END_STEP}" \
  algo.max_noise_level=1.0 \
  algo.clean_loss_weight="${CLEAN_LOSS_WEIGHT}" \
  algo.logit_kl_weight=0.0 \
  algo.gradient_cosine_every=0 \
  algo.temp=0.1 \
  algo.rolling_warmup_steps=8 \
  algo.rolling_refinement_steps=1 \
  algo.rolling_x0_projection=argmax \
  wandb.project="${WANDB_PROJECT}" \
  wandb.group="${WANDB_GROUP}" \
  wandb.name="${RUN_NAME}" \
  wandb.id="${RUN_NAME}-seed${SEED}" \
  "wandb.tags=[openwebtext,rolling-diff,block${BLOCK_SIZE},sequence-constant-noise,no-kl]" \
  hydra.run.dir="${RUN_DIR}" \
  checkpointing.save_dir="${RUN_DIR}" \
  checkpointing.resume_ckpt_path="${RUN_DIR}/checkpoints/last.ckpt"

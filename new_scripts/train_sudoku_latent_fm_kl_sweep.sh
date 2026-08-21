#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sweep_root="${repo_root}/new_sudoku_runs/latent_fm_kl_sweep"
gpus=(2)
betas=(0.00001 0.0001 0.001)
max_steps=${LATENT_FM_SWEEP_STEPS:-5000}

mkdir -p "${sweep_root}"
pids=()
labels=()
status=0

for index in "${!betas[@]}"; do
  if (( index >= ${#gpus[@]} )); then
    slot=$((index - ${#gpus[@]}))
    if ! wait "${pids[slot]}"; then
      echo "LATENT_FM_KL_FAILED:${labels[slot]}" >&2
      status=1
    fi
  fi

  beta=${betas[index]}
  label=${beta//./p}
  gpu=${gpus[index%${#gpus[@]}]}
  run_dir="${sweep_root}/beta_${label}"
  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
    --config-name=config \
    +experiment=sudoku_latent_fm \
    "algo.kl_beta=${beta}" \
    "trainer.max_steps=${max_steps}" \
    wandb.project=sudoku-latent-fm-kl-sweep \
    "wandb.name=latent-fm-sequence-kl-beta-${beta}-5k" \
    checkpointing.resume_from_ckpt=false \
    "checkpointing.save_dir=${run_dir}" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
    callbacks.sudoku_eval.every_n_train_steps=1000 \
    callbacks.sudoku_eval.full_eval_at_end=false \
    +trainer.check_val_every_n_epoch=null \
    "hydra.run.dir=${run_dir}" \
    >"${run_dir}/train.log" 2>&1 &
  pids+=("$!")
  labels+=("beta_${label}")
done

completed_slots=$((${#betas[@]} - ${#gpus[@]}))
for index in "${!pids[@]}"; do
  if (( index < completed_slots )); then
    continue
  fi
  if wait "${pids[index]}"; then
    echo "LATENT_FM_KL_OK:${labels[index]}"
  else
    echo "LATENT_FM_KL_FAILED:${labels[index]}" >&2
    status=1
  fi
done
exit "${status}"

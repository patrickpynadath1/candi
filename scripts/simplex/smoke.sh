#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${OUTPUT:-$ROOT/runs/simplex-smoke}"
for method in simplex mdlm uniform; do
  for sc in false true; do
    experiment=sudoku_simplex
    if [[ "$method" != simplex ]]; then experiment="sudoku_simplex_$method"; fi
    run="$OUTPUT/${method}_sc${sc}"
    "$PYTHON_BIN" "$ROOT/main.py" "+experiment=$experiment" \
      "self_conditioning.enabled=$sc" \
      model.hidden_size=16 model.cond_dim=8 model.n_heads=2 model.n_blocks=1 \
      data.num_train=4 data.num_valid=2 "data.cache_dir=$OUTPUT/data_cache" \
      loader.global_batch_size=2 loader.eval_global_batch_size=2 loader.num_workers=0 \
      trainer.accelerator=cpu trainer.devices=1 trainer.accumulate_grad_batches=1 \
      trainer.max_steps=2 trainer.val_check_interval=2 trainer.limit_val_batches=1 \
      trainer.num_sanity_val_steps=0 trainer.log_every_n_steps=1 \
      strategy._target_=lightning.pytorch.strategies.SingleDeviceStrategy \
      +strategy.device=cpu '~strategy.find_unused_parameters' \
      callbacks.sudoku_eval.every_n_train_steps=2 callbacks.sudoku_eval.subset_size=2 \
      callbacks.sudoku_eval.batch_size=2 callbacks.sudoku_eval.num_steps=3 \
      callbacks.checkpoint_every_n_steps.every_n_train_steps=2 \
      sampling.steps=3 "hydra.run.dir=$run" "checkpointing.save_dir=$run" \
      "wandb.project=${WANDB_PROJECT:-simplex-sudoku}" "wandb.name=smoke_${method}_sc${sc}" \
      wandb.job_type=smoke wandb.mode=online checkpointing.resume_from_ckpt=false
  done
done

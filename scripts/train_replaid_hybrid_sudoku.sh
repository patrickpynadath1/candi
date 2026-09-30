#!/usr/bin/env bash
# Run from a frozen source snapshot; the caller supplies CUDA_VISIBLE_DEVICES.
set -euo pipefail
run_dir=$1
python_bin=$2
data_dir=$3
trap 'echo failed > "$run_dir/status.txt"' ERR
export WANDB_MODE=online
echo training > "$run_dir/status.txt"
"$python_bin" main.py +experiment=sudoku_replaid_hybrid_matched \
  "data.cache_dir=$data_dir" \
  "wandb.id=$(basename "$run_dir")" \
  "hydra.run.dir=$run_dir" "checkpointing.save_dir=$run_dir" \
  +trainer.enable_progress_bar=false > "$run_dir/train.log" 2>&1

echo evaluating > "$run_dir/status.txt"
"$python_bin" main.py +experiment=sudoku_replaid_hybrid_matched mode=sudoku_eval \
  "data.cache_dir=$data_dir" \
  "eval.checkpoint_path=$run_dir/checkpoints/last.ckpt" \
  "hydra.run.dir=$run_dir/eval10k" "sudoku.output_dir=$run_dir/eval10k" \
  sampling.steps=180 > "$run_dir/eval.log" 2>&1
echo complete > "$run_dir/status.txt"

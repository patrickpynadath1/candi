#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
algorithms=(mdlm candi)
gpus=(6 7)

for index in "${!algorithms[@]}"; do
  algorithm=${algorithms[index]}
  gpu=${gpus[index]}
  run_dir="${repo_root}/new_smoke_runs/sudoku_hard/${algorithm}"
  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
    --config-name=config \
    "+experiment=sudoku_${algorithm}" \
    data.difficulty=hard \
    data.num_train=8 \
    data.num_valid=2 \
    data.sudoku_num_workers=1 \
    +trainer.fast_dev_run=true \
    trainer.accumulate_grad_batches=1 \
    loader.global_batch_size=2 \
    loader.eval_global_batch_size=2 \
    loader.num_workers=0 \
    checkpointing.resume_from_ckpt=false \
    "checkpointing.save_dir=${run_dir}" \
    "hydra.run.dir=${run_dir}" \
    wandb=null \
    >"${run_dir}/smoke.log" 2>&1
  echo "SUDOKU_HARD_SMOKE_OK:${algorithm}"
done

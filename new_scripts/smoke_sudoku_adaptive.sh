#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
algorithms=(mdlm ar candi duo flm sfm)
gpus=(5 6 7)
smoke_root="${repo_root}/new_sudoku_runs/adaptive_smoke"
mkdir -p "${smoke_root}"
pids=()
names=()

for index in "${!algorithms[@]}"; do
  algorithm=${algorithms[index]}
  gpu=${gpus[index%${#gpus[@]}]}
  experiment="sudoku_${algorithm}_adaptive"
  if [[ "${algorithm}" == "ar" ]]; then
    experiment=sudoku_ar_adaptive_control
  fi
  run_dir="${smoke_root}/${algorithm}"
  mkdir -p "${run_dir}"
  overrides=()
  if [[ "${algorithm}" != "ar" ]]; then
    overrides+=(
      noise.adaptive_buffer_size=2
      noise.adaptive_refit_every=1
      noise.adaptive_warmup_steps=0
      noise.adaptive_n_grid=32
      noise.adaptive_n_knots=4
      noise.adaptive_spline_degree=2
    )
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
    --config-name=config \
    "+experiment=${experiment}" \
    +trainer.fast_dev_run=true \
    trainer.accumulate_grad_batches=1 \
    loader.global_batch_size=2 \
    wandb=null \
    "checkpointing.save_dir=${run_dir}" \
    "hydra.run.dir=${run_dir}" \
    "${overrides[@]}" >"${run_dir}/smoke.log" 2>&1 &
  pids+=("$!")
  names+=("${algorithm}")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    echo "SUDOKU_ADAPTIVE_SMOKE_OK:${names[index]}"
  else
    echo "SUDOKU_ADAPTIVE_SMOKE_FAILED:${names[index]}" >&2
    status=1
  fi
done
exit "${status}"

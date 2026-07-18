#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
algorithms=(mdlm ar mdlm_perm candi cadd duo embedding_diffusion cont_diff_onehot semicat)
gpus=(5 6 7)
smoke_root="${repo_root}/new_smoke_runs"
mkdir -p "${smoke_root}"

run_smoke() {
  local gpu=$1
  local algorithm=$2
  local run_dir="${smoke_root}/${algorithm}"
  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
    --config-name=config_text8_frontier \
    "algo=${algorithm}" \
    wandb=null \
    +trainer.fast_dev_run=true \
    trainer.accumulate_grad_batches=1 \
    loader.global_batch_size=8 \
    loader.eval_global_batch_size=8 \
    loader.num_workers=0 \
    sampling.steps=2 \
    checkpointing.resume_from_ckpt=false \
    "checkpointing.save_dir=${run_dir}" \
    "hydra.run.dir=${run_dir}" \
    >"${run_dir}/smoke.log" 2>&1
}

pids=()
names=()
status=0
for index in "${!algorithms[@]}"; do
  algorithm=${algorithms[index]}
  gpu=${gpus[index%${#gpus[@]}]}
  run_smoke "${gpu}" "${algorithm}" &
  pids+=("$!")
  names+=("${algorithm}")
  if [[ ${#pids[@]} -eq ${#gpus[@]} ]]; then
    for job in "${!pids[@]}"; do
      if wait "${pids[job]}"; then
        echo "SMOKE_OK:${names[job]}"
      else
        echo "SMOKE_FAILED:${names[job]}" >&2
        status=1
      fi
    done
    pids=()
    names=()
  fi
done
for job in "${!pids[@]}"; do
  if wait "${pids[job]}"; then
    echo "SMOKE_OK:${names[job]}"
  else
    echo "SMOKE_FAILED:${names[job]}" >&2
    status=1
  fi
done
exit "${status}"

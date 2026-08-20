#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 [--nfe N] GPU_ID [GPU_ID ...]" >&2
  exit 2
fi

nfe=64
if [[ ${1:-} == "--nfe" ]]; then
  if [[ $# -lt 3 ]]; then
    echo "--nfe requires a value followed by at least one GPU id" >&2
    exit 2
  fi
  nfe=$2
  shift 2
fi
gpus=("$@")
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
run_root="${storage_root}/new_sudoku_runs/blurring"
launcher_root="${storage_root}/new_sudoku_runs/blurring_greedy_eval/nfe_${nfe}"
mkdir -p "${launcher_root}"

jobs=(
  dct_d2:2:dct:live
  dct_d4:4:dct:live
  dct_d8:8:dct:live
  dct_d16:16:dct:live
  isotropic_d2:2:isotropic:live
  isotropic_d4:4:isotropic:live
  isotropic_d8:8:isotropic:live
  isotropic_d16:16:isotropic:live
  dct_d4_detached:4:dct:detached
)

run_queue() {
  local worker_index=$1 gpu_id=${gpus[$1]}
  local index spec label input_dim basis embedding_mode checkpoint output_dir detach
  for ((index=worker_index; index<${#jobs[@]}; index+=${#gpus[@]})); do
    spec=${jobs[index]}
    IFS=: read -r label input_dim basis embedding_mode <<<"${spec}"
    checkpoint="${run_root}/${label}/checkpoints/last.ckpt"
    output_dir="${run_root}/${label}/evaluation_greedy/nfe_${nfe}"
    detach=false
    [[ ${embedding_mode} == detached ]] && detach=true
    if [[ ! -f ${checkpoint} ]]; then
      echo "missing checkpoint: ${checkpoint}" >&2
      return 1
    fi
    mkdir -p "${output_dir}"
    echo "[$(date +%H:%M:%S)] GPU ${gpu_id}: evaluating ${label} at ${nfe} NFE"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
      --config-name=config \
      +experiment=sudoku_blurring_hard \
      mode=sudoku_eval \
      "data.cache_dir=${storage_root}/data_cache" \
      "algo.input_embedding_dim=${input_dim}" \
      "algo.basis=${basis}" \
      "algo.detach_forward_embedding=${detach}" \
      sampling.greedy_clean_estimate=true \
      "sampling.steps=${nfe}" \
      sudoku.batch_size=64 \
      loader.eval_global_batch_size=64 \
      loader.num_workers=4 \
      "eval.checkpoint_path=${checkpoint}" \
      "sudoku.output_dir=${output_dir}" \
      "hydra.run.dir=${output_dir}" \
      wandb=null \
      >"${launcher_root}/${label}.log" 2>&1
    echo "[$(date +%H:%M:%S)] GPU ${gpu_id}: finished ${label}"
  done
}

pids=()
for index in "${!gpus[@]}"; do
  run_queue "${index}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ ${status} -ne 0 ]]; then
  echo "one or more evaluations failed; inspect ${launcher_root}" >&2
  exit "${status}"
fi
echo "all greedy evaluations completed: ${launcher_root}"

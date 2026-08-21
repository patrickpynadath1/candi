#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then echo "usage: $0 GPU_ID LABEL" >&2; exit 2; fi
gpu_id=$1
label=$2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
result_root="${storage_root}/new_eval_results/sudoku_hard_eb_0p1_greedy"
output_dir="${result_root}/${label}"
mkdir -p "${output_dir}"
extra=()
case "${label}" in
  candi_blurred_d16) experiment=sudoku_candi_blurred_hard_eb; checkpoint="${storage_root}/new_sudoku_runs/candi_blurred_d16_hard_eb/checkpoints/last.ckpt"; extra+=(algo.input_embedding_dim=16) ;;
  candi_blurred_d512) experiment=sudoku_candi_blurred_hard_eb; checkpoint="${storage_root}/new_sudoku_runs/candi_blurred_d512_hard_eb/checkpoints/last.ckpt"; extra+=(algo.input_embedding_dim=512) ;;
  candi) experiment=sudoku_candi_hard_eb; checkpoint="${storage_root}/new_sudoku_runs/hard_eb_0p1/candi/checkpoints/last.ckpt" ;;
  mdlm) experiment=sudoku_mdlm_hard_eb; checkpoint="${storage_root}/new_sudoku_runs/hard_eb_0p1/mdlm/checkpoints/last.ckpt" ;;
  *) echo "unknown evaluation: ${label}" >&2; exit 2 ;;
esac
[[ -f "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 1; }
CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
  --config-name=config "+experiment=${experiment}" mode=sudoku_eval \
  "data.cache_dir=${storage_root}/data_cache" \
  sampling.entropy_bound=0.1 sampling.greedy_clean_estimate=true \
  sampling.steps=180 sudoku.max_examples=2000 sudoku.batch_size=64 \
  loader.eval_global_batch_size=64 loader.num_workers=1 \
  "eval.checkpoint_path=${checkpoint}" "sudoku.output_dir=${output_dir}" \
  "hydra.run.dir=${output_dir}" "${extra[@]}" wandb=null \
  >"${result_root}/${label}.log" 2>&1

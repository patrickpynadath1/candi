#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "usage: $0 [MDLM_GPU] [CANDI_GPU]" >&2
  exit 2
fi

mdlm_gpu=${1:-6}
candi_gpu=${2:-7}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}

"${repo_root}/.venv/bin/python" "${repo_root}/new_scripts/prepare_sudoku.py" \
  --cache-dir "${storage_root}/data_cache" \
  --difficulty hard \
  --num-train 48000 \
  --num-valid 2000 \
  --workers 32

"${repo_root}/new_scripts/launch_sudoku_hard_mdlm_candi_20k.sh" \
  "${mdlm_gpu}" "${candi_gpu}"

#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DUO_STORAGE_ROOT:-/home/patrick/.cache/duo}
directories=(new_sudoku_runs new_smoke_runs new_eval_results)

mkdir -p "${storage_root}"
for name in "${directories[@]}"; do
  source_path="${repo_root}/${name}"
  destination_path="${storage_root}/${name}"
  if [[ -L "${source_path}" ]]; then
    echo "already migrated: ${source_path} -> $(readlink "${source_path}")"
    continue
  fi
  if [[ ! -d "${source_path}" ]]; then
    echo "missing source directory: ${source_path}" >&2
    exit 1
  fi
  if [[ -e "${destination_path}" ]]; then
    echo "destination already exists: ${destination_path}" >&2
    exit 1
  fi
  echo "moving ${source_path} to ${destination_path}"
  mv "${source_path}" "${destination_path}"
  ln -s "${destination_path}" "${source_path}"
  [[ -d "${source_path}" ]]
  echo "migrated: ${source_path} -> ${destination_path}"
done

#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
"${repo_root}/new_scripts/train_all.sh"
"${repo_root}/new_scripts/eval_all.sh"

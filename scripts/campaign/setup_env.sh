#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
: "${UV_PROJECT_ENVIRONMENT:=$root/.venv}"
export UV_PROJECT_ENVIRONMENT
uv sync --locked --extra tuning --extra scoring --extra tracking
"$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import torch, ray; print("torch", torch.__version__, "ray", ray.__version__, "CUDA available", torch.cuda.is_available())'
echo "Set CANDI_PYTHON=$UV_PROJECT_ENVIRONMENT/bin/python on every allocated node."

#!/usr/bin/env bash
set -euo pipefail
: "${CANDI_ROOT:?Absolute shared source directory required}"
: "${CANDI_PYTHON:?Absolute shared environment Python required}"
cd "$CANDI_ROOT"
export PYTHONPATH="$CANDI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$CANDI_PYTHON" -m core.cluster.slurm submit "$@"

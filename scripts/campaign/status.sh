#!/usr/bin/env bash
set -euo pipefail
: "${CANDI_ROOT:?Absolute shared source directory required}"
: "${CANDI_PYTHON:?Absolute shared environment Python required}"
cd "$CANDI_ROOT"
exec "$CANDI_PYTHON" -m core.cluster.slurm status "$@"

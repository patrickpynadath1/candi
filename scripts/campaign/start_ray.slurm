#!/usr/bin/env bash
# Supply site-specific resources to sbatch; no account/partition is hard-coded.
#SBATCH --job-name=candi-ray
#SBATCH --ntasks-per-node=1
set -euo pipefail
: "${CANDI_ROOT:?Absolute shared source directory required}"
: "${CANDI_PYTHON:?Absolute shared environment Python required}"
: "${CANDI_CLUSTER_DIR:?New shared directory for this allocation required}"
: "${CANDI_GPUS_PER_NODE:?Number of allocated GPUs per node required}"
: "${CANDI_CPUS_PER_NODE:?Number of allocated CPUs per node required}"
cd "$CANDI_ROOT"
export PYTHONPATH="$CANDI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$CANDI_PYTHON" -m core.cluster.slurm start \
  --directory "$CANDI_CLUSTER_DIR" --gpus "$CANDI_GPUS_PER_NODE" \
  --cpus "$CANDI_CPUS_PER_NODE" "${@}"

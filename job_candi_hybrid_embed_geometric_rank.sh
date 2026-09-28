#!/usr/bin/env bash
#SBATCH --job-name=tinygsm-hybrid-embed-geometric-rank
#SBATCH --partition=ai
#SBATCH --account=ruqiz
#SBATCH --qos=preemptible
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
# Submit from the repository root (mkdir -p logs first), or set CANDI_ROOT.
repo_root=${CANDI_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}
exec bash "${repo_root}/scripts/run_candi_hybrid_embedding.sh" geometric_rank "$@"

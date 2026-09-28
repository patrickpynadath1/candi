#!/usr/bin/env bash
#SBATCH --job-name=tinygsm-block64-linear-no-noise-warmup
#SBATCH --partition=ai
#SBATCH --account=ruqiz
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
# Full noise range from step zero; retain the learning-rate warmup.
repo_root=${CANDI_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}
export RUN_NAME=tinygsm-block64-embed-linear-no-noise-warmup
exec bash "${repo_root}/scripts/run_block_embedding.sh" linear \
  algo.noise_curriculum_start_step=0 algo.noise_curriculum_end_step=0 "$@"

#!/usr/bin/env bash
#SBATCH --job-name=tinygsm-blockgen-masked
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
# Submit from the repository root, or set CANDI_ROOT.
repo_root=${CANDI_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}
cd "$repo_root"
python_bin=${CANDI_PYTHON:-${UV_PROJECT_ENVIRONMENT:-$HOME/candi-1/.venv}/bin/python}
[[ -x "$python_bin" ]] || { echo "Missing Python environment: $python_bin" >&2; exit 1; }
scratch_root=${SCRATCH_DIR:-${RCAC_SCRATCH:?Set SCRATCH_DIR or RCAC_SCRATCH}}
run_name=tinygsm-blockgen-masked
run_id=${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}
run_dir=${RUN_DIR:-$scratch_root/cont_diff/candiv2.1/$run_name/$run_id}
cache_dir=${DATA_CACHE_DIR:-$scratch_root/candi-1-data/tinygsm}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
args=(
  +experiment=tinygsm_blockgen_masked strategy=ddp
  "data.cache_dir=$cache_dir" "hydra.run.dir=$run_dir" "output_dir=$run_dir"
  '+wandb.project=candi-tinygsm' '+wandb.entity=rz-text-dlp'
  '+wandb.group=blockgen-comparison' "+wandb.name=$run_name"
  "+wandb.id=blockgen-$run_id" '+wandb.resume=allow'
)
for arg in "$@"; do
  case "$arg" in
    --cfg|--cfg=*|--help) exec "$python_bin" main.py "${args[@]}" "$@" ;;
  esac
done
mkdir -p "$run_dir" "$cache_dir"
# Serialize initial cache preparation across the two jobs; their data recipe is identical.
flock "$cache_dir/.blockgen-prepare.lock" "$python_bin" - "${args[@]}" "$@" <<'PYCODE'
import sys
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
    config = compose(config_name="config", overrides=sys.argv[1:])
    instantiate(config.task).prepare_data()
PYCODE
resume_args=()
if (( ${SLURM_RESTART_COUNT:-0} > 0 )) && [[ -f "$run_dir/checkpoints/last.ckpt" ]]; then
  resume_args=("checkpoint.resume=$run_dir/checkpoints/last.ckpt")
fi
exec "$python_bin" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node=8 main.py "${args[@]}" "$@" "${resume_args[@]}"

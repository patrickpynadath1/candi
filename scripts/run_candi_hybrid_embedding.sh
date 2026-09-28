#!/usr/bin/env bash
# Shared launch settings for the four job_candi_hybrid_embed_*.sh jobs.
set -euo pipefail
variant=${1:?Expected linear, geometric_rank, linear_rank, or adaptive_linear_rank}
shift
case "$variant" in
  linear|geometric_rank|linear_rank) interpolation=$variant; adaptive=false ;;
  adaptive_linear_rank) interpolation=linear_rank; adaptive=true ;;
  *) echo "Unknown embedding schedule: $variant" >&2; exit 2 ;;
esac
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

# Reuse the installed uv environment; do not resolve/install packages on workers.
if [[ -z ${UV_PROJECT_ENVIRONMENT:-} ]]; then
  for candidate in "$repo_root/.venv" "$HOME/candi-1/.venv"; do
    if [[ -x "$candidate/bin/python" ]]; then UV_PROJECT_ENVIRONMENT=$candidate; break; fi
  done
fi
: "${UV_PROJECT_ENVIRONMENT:?Set UV_PROJECT_ENVIRONMENT to an installed candi-v2 uv environment}"
export UV_PROJECT_ENVIRONMENT
uv_bin=${UV_BIN:-$(command -v uv || true)}
if [[ -z $uv_bin && -x $HOME/.local/bin/uv ]]; then uv_bin=$HOME/.local/bin/uv; fi
: "${uv_bin:?Install uv or set UV_BIN to its executable}"
runner=("$uv_bin" run --no-sync)
scratch_dir=${SCRATCH_DIR:-${RCAC_SCRATCH:?Set SCRATCH_DIR or RCAC_SCRATCH}/cont_diff}
nproc=${NPROC_PER_NODE:-8}
run_name="tinygsm-hybrid-embed-${variant}"
run_dir="${RUN_DIR:-${scratch_dir}/candi-v2/${run_name}/${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

args=(
  +experiment=tinygsm_candi algo=candi_embedding model=small model.length=512
  algo.is_embed=true algo.pure_continuous=false
  +algo.legacy_architecture=false
  algo.continuous_path=flow_matching algo.hybrid_loss_weighting=none
  "algo.interpolation_schedule=$interpolation"
  algo.min_percentile=0.01 algo.max_percentile=0.49 algo.rank_endpoint_width=0.0
  noise=rank-adaptive "noise.adaptive=$adaptive" noise.eps=0.0
  sampling.predictor=candi_embedding sampling.steps=32 eval.generate_samples=false
  "scratch_dir=$scratch_dir"
  "data.cache_dir=${DATA_CACHE_DIR:-${RCAC_SCRATCH:?Set RCAC_SCRATCH or DATA_CACHE_DIR}/candi-1-data/tinygsm}"
  "trainer.devices=$nproc" trainer.num_nodes=1
  trainer.max_steps=250000 trainer.accumulate_grad_batches=4 batch_size=512
  'loader.global_batch_size=${batch_size}'
  trainer.val_check_interval=5000 +trainer.check_val_every_n_epoch=null
  callbacks.gsm8k_eval.every_n_train_steps=5000
  callbacks.gsm8k_eval.full_eval_at_end=false
  optim.lr=0.0003 lr_scheduler=constant_warmup lr_scheduler.num_warmup_steps=2500
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000
  checkpointing.resume_from_ckpt=false "checkpointing.save_dir=$run_dir"
  "hydra.run.dir=$run_dir" wandb.project=candi-tinygsm
  "wandb.name=$run_name" wandb.group=hybrid-embedding-rank
)

# Hydra configuration inspection needs neither GPUs nor distributed workers.
for arg in "$@"; do
  if [[ $arg == --cfg || $arg == --cfg=* || $arg == --help ]]; then
    exec "${runner[@]}" python main.py "${args[@]}" "$@"
  fi
done
restart_args=()
if (( ${SLURM_RESTART_COUNT:-0} > 0 )); then
  save_dir=$run_dir
  fallback=""
  for arg in "$@"; do
    case "$arg" in
      checkpointing.save_dir=*) save_dir=${arg#*=} ;;
      checkpointing.resume_ckpt_path=*) fallback=${arg#*=} ;;
    esac
  done
  latest=$("${runner[@]}" python scripts/latest_training_checkpoint.py "$save_dir" --fallback "$fallback")
  echo "Slurm restart ${SLURM_RESTART_COUNT}: resuming $latest" >&2
  restart_args=(checkpointing.resume_from_ckpt=true "checkpointing.resume_ckpt_path=$latest")
fi
exec "${runner[@]}" python -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="$nproc" main.py "${args[@]}" "$@" "${restart_args[@]}"

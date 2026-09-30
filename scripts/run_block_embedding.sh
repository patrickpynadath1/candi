#!/usr/bin/env bash
# TinyGSM block diffusion: monotone position clocks, block size 64.
set -euo pipefail
variant=${1:?Expected linear or linear_rank}
shift
case "$variant" in
  linear|linear_rank) ;;
  *) echo "Unknown block embedding schedule: $variant" >&2; exit 2 ;;
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
run_name="${RUN_NAME:-tinygsm-block64-embed-${variant}}"
run_dir="${RUN_DIR:-${scratch_dir}/candi-v2/${run_name}/${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

args=(
  +experiment=tinygsm_block_embedding model=small model.length=512
  algo.block_size=64 algo.noise_level_mode=monotone
  "algo.interpolation_schedule=$variant" noise.adaptive=false
  algo.min_percentile=0.01 algo.max_percentile=0.49 algo.rank_endpoint_width=0.01
  algo.noise_curriculum_start_step=2500 algo.noise_curriculum_end_step=5000
  sampling.predictor=block_diffusion_euler eval.generate_samples=false
  "scratch_dir=$scratch_dir"
  "data.cache_dir=${DATA_CACHE_DIR:-${RCAC_SCRATCH:?Set RCAC_SCRATCH or DATA_CACHE_DIR}/candi-1-data/tinygsm}"
  "trainer.devices=$nproc" trainer.num_nodes=1
  trainer.max_steps=250000 trainer.accumulate_grad_batches=4 batch_size=512
  'loader.global_batch_size=${batch_size}' loader.multiprocessing_context=spawn
  trainer.val_check_interval=5000 +trainer.check_val_every_n_epoch=null
  callbacks.gsm8k_eval.every_n_train_steps=5000 callbacks.gsm8k_eval.full_eval_at_end=false
  optim.lr=0.0003 lr_scheduler=constant_warmup lr_scheduler.num_warmup_steps=2500
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000
  checkpointing.resume_from_ckpt=false "checkpointing.save_dir=$run_dir"
  "hydra.run.dir=$run_dir" wandb.project=candi-tinygsm
  "wandb.name=$run_name" wandb.group=block64-embedding-schedules
)
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  args+=("wandb.id=block64-${SLURM_JOB_ID}" +wandb.resume=allow)
fi

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
  if compgen -G "$save_dir/checkpoints/*.ckpt" >/dev/null || [[ -f $fallback ]]; then
    latest=$("${runner[@]}" python scripts/latest_training_checkpoint.py "$save_dir" --fallback "$fallback")
    echo "Slurm restart ${SLURM_RESTART_COUNT}: resuming $latest" >&2
    restart_args=(checkpointing.resume_from_ckpt=true "checkpointing.resume_ckpt_path=$latest")
  else
    echo "Requeued before a checkpoint was saved; starting fresh." >&2
  fi
fi
exec "${runner[@]}" python -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="$nproc" main.py "${args[@]}" "$@" "${restart_args[@]}"

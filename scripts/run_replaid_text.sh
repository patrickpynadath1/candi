#!/usr/bin/env bash
# RePlaid and full hybrid ELBO training on the existing Gautschi caches.
set -euo pipefail
dataset=${1:?Expected tinygsm or openwebtext}
variant=${2:?Expected replaid or replaid_hybrid}
shift 2
case "$dataset" in tinygsm|openwebtext) ;; *) exit 2 ;; esac
case "$variant" in replaid|replaid_hybrid) ;; *) exit 2 ;; esac
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
run_name="${RUN_NAME:-${dataset}-${variant}}"
run_dir="${RUN_DIR:-${scratch_dir}/candi-v2/${run_name}/${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

args=(
  "algo=$variant" model=small
  "sampling.predictor=${variant}_ddpm" sampling.steps=32 training.sampling_eps=0.0
  algo.clean_context=true eval.generate_samples=false
  "scratch_dir=$scratch_dir"
  "trainer.devices=$nproc" trainer.num_nodes=1 trainer.precision=bf16-mixed
  trainer.max_steps=250000 batch_size=512 'loader.global_batch_size=${batch_size}'
  loader.multiprocessing_context=spawn loader.num_workers=4
  loader.eval_global_batch_size=128 trainer.limit_val_batches=8
  trainer.num_sanity_val_steps=2 +trainer.check_val_every_n_epoch=null
  optim.lr=0.0003 optim.weight_decay=0.0
  lr_scheduler=constant_warmup lr_scheduler.num_warmup_steps=2500
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000
  checkpointing.resume_from_ckpt=false "checkpointing.save_dir=$run_dir"
  "hydra.run.dir=$run_dir" "wandb.project=candi-${dataset}"
  "wandb.name=$run_name" wandb.group=replaid-comparison +wandb.entity=rz-text-dlp
)
if [[ $dataset == tinygsm ]]; then
  args+=(+experiment=tinygsm_candi "algo=$variant" model.length=512
    trainer.accumulate_grad_batches=8 trainer.val_check_interval=40000
    "data.cache_dir=${DATA_CACHE_DIR:-${RCAC_SCRATCH}/candi-1-data/tinygsm}"
    callbacks.gsm8k_eval.every_n_train_steps=5000
    callbacks.gsm8k_eval.full_eval_at_end=false callbacks.gsm8k_eval.batch_size=4
    "callbacks.gsm8k_eval.output_dir=$run_dir/gsm8k_eval")
else
  args+=(data=openwebtext model.length=1024
    trainer.accumulate_grad_batches=16 trainer.val_check_interval=80000
    "data.cache_dir=${DATA_CACHE_DIR:-${scratch_dir}/owt}")
fi
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  args+=("wandb.id=replaid-${SLURM_JOB_ID}" +wandb.resume=allow)
fi
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

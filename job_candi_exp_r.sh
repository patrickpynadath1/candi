#!/bin/bash
#SBATCH -J owt-candi-exp-r
#SBATCH -p gpu                 # H100 partition  :contentReference[oaicite:0]{index=0}
#SBATCH --nodes=1              # keep all GPUs on the same node
#SBATCH --gres=gpu:8           # all 8 H100S
#SBATCH -A ruqiz
#SBATCH --cpus-per-task=112    # give the single task every core
#SBATCH -t 15:00:00              # wall-time (edit as needed)
#SBATCH -o logs/%x-%j.out      # capture stdout/stderr

module purge
module load conda
conda activate candi

export SCRATCH_DIR=$RCAC_SCRATCH/cont_diff
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=8 main.py scratch_dir=$SCRATCH_DIR \
	data=openwebtext \
	wandb.name=candi-.01-.25-exp-r \
	optim.lr=.0003 \
	lr_scheduler=constant_warmup \
	lr_scheduler.num_warmup_steps=2500 \
	wandb.project=candi-owt-rebuttal \
	algo=candi \
	model=small \
	model.length=1024 \
	trainer.max_steps=50000 \
	trainer.accumulate_grad_batches=4 \
	batch_size=512 \
	trainer.val_check_interval=5000 \
	algo.cont_schedule_type=geometric \
	algo.cont_schedule_target=r \
	algo.mixed_coeff=0.5 \
	algo.use_percentile_scheduling=True \
	algo.min_percentile=.01 \
	algo.max_percentile=.25;

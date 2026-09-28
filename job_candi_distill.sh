#!/bin/bash
#SBATCH -J owt-candi
#SBATCH -p gpu                 # H100 partition  :contentReference[oaicite:0]{index=0}
#SBATCH --nodes=1              # keep all GPUs on the same node
#SBATCH --gres=gpu:8           # all 8 H100
#SBATCH -A ruqiz
#SBATCH --cpus-per-task=112    # give the single task every core
#SBATCH -t 48:00:00              # wall-time (edit as needed)
#SBATCH -o logs/%x-%j.out      # capture stdout/stderr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ppynadat@purdue.edu


module purge
module load conda
conda activate cont_diff

export SCRATCH_DIR=$RCAC_SCRATCH/cont_diff
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=8 main.py scratch_dir=$SCRATCH_DIR \
	data=openwebtext \
	wandb.name=distill-bwd-kl-high-lr \
	optim.lr=0.00006 \
	lr_scheduler=constant_warmup \
	lr_scheduler.num_warmup_steps=500 \
	wandb.project=candi-owt \
	algo=candi_distill \
	checkpointing.resume_ckpt_path="/scratch/gautschi/ppynadat/cont_diff/openwebtext-train/2025.09.15/221843/checkpoints/44-735500.ckpt" \
	model=small \
	model.length=1024 \
	trainer.max_steps=2000000 \
	trainer.accumulate_grad_batches=4 \
	batch_size=128 \
	trainer.val_check_interval=10000 \
	algo.weighted_loss=False \
	algo.clean_reveal_ratio=1 \
	algo.use_mixed_mask=True \
	algo.mixed_coeff=0.5 \
	algo.use_percentile_scheduling=True \
	algo.min_percentile=.01 \
	algo.max_percentile=.25 \
	algo.number_steps=128 \
	algo.adapt_every=20000 \
	algo.budget=100000 \
	algo.sampler=normal \
	trainer.limit_val_batches=0.05 \
	algo.temp=.9;

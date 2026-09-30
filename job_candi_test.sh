#!/bin/bash

module purge
module load conda
conda activate candi

export SCRATCH_DIR=$RCAC_SCRATCH/cont_diff
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=1 main.py scratch_dir=$SCRATCH_DIR \
	data=openwebtext \
	wandb.name=candi-test \
	optim.lr=.0003 \
	lr_scheduler=constant_warmup \
	lr_scheduler.num_warmup_steps=2500 \
	wandb.project=testing \
	algo=candi \
	model=small \
	model.length=1024 \
	trainer.max_steps=400000 \
	trainer.accumulate_grad_batches=32 \
	batch_size=512 \
	trainer.val_check_interval=5000 \
	algo.mixed_coeff=0.5 \
	algo.use_percentile_scheduling=True \
	algo.min_percentile=.05 \
	algo.max_percentile=.4;

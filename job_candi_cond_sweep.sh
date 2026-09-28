#!/bin/bash
#SBATCH -J candi-cond-eval
#SBATCH -p gpu                 # H100 partition  :contentReference[oaicite:0]{index=0}
#SBATCH --nodes=1              # keep all GPUs on the same node
#SBATCH --ntasks-per-node=1    # One task per node (important for multi-node)
#SBATCH --gres=gpu:8           # all 8 H100
#SBATCH -A ruqiz
#SBATCH --cpus-per-task=112    # give the single task every core
#SBATCH -t 00:04:00              # wall-time (edit as needed)
#SBATCH -o logs/%x-%j.out      # capture stdout/stderr

module purge
module load conda
conda activate cont_diff

export SCRATCH_DIR=$RCAC_SCRATCH/cont_diff
torchrun --nproc_per_node=8 main_for_sweeps_ddp.py --config-name=conditional_eval \
	mode=conditional_sweep \
	scratch_dir=$SCRATCH_DIR \
	eval.num_conditional_samples=5000 \
	data=openwebtext \
	model=small \
	algo=candi \
	model.length=1024 \
	batch_size=256 \
	loader.eval_global_batch_size=64 \
	algo.clean_reveal_ratio=1 \
	algo.use_mixed_mask=True \
	algo.mixed_coeff=0.5 \
	eval.revealed_fraction=0.05 \
	algo.use_percentile_scheduling=True \
	eval.checkpoint_path="/scratch/gautschi/ppynadat/cont_diff/candi_eval_ckpt/candi.ckpt"

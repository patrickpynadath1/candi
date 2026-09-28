#!/bin/bash
#SBATCH -J owt-candi-gen-full
#SBATCH -p gpu                 # H100 partition  :contentReference[oaicite:0]{index=0}
#SBATCH --nodes=1              # keep all GPUs on the same node
#SBATCH --gres=gpu:1           # all 8 H100S
#SBATCH -A ruqiz
#SBATCH --cpus-per-task=14    # give the single task every core
#SBATCH -t 1:00:00              # wall-time (edit as needed)
#SBATCH -o logs/%x-%j.out      # capture stdout/stderr

module purge
module load conda
conda activate candi

export HYDRA_FULL_ERROR=1
export SCRATCH_DIR=$RCAC_SCRATCH/cont_diff

eval_batch_size=8
mixed_coeff=.5
CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py scratch_dir=$SCRATCH_DIR \
  mode=sample_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=32 \
  data=openwebtext-split \
  model=small \
  algo=candi \
  sampling.num_sample_batches=4 \
  eval.checkpoint_path=$RCAC_SCRATCH/cont_diff/openwebtext-train/candi-orig-full/best.ckpt \
  eval.generated_samples_path=$RCAC_SCRATCH/cont_diff/gen-owt/candi-full-rerun/samples.json \
  model.length=1024 \
  algo.step_size=1 \
  algo.sampler=cached \
  algo.mixed_coeff=$mixed_coeff \
  +wandb.offline=true;


  # distill-r3-bwd
  # eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/openwebtext-train/owt-1024-latest5/checkpoints/best.ckpt \

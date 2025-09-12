#!/bin/bash
#SBATCH -J sample_mdlm                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption


export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=conditional_eval \
  mode=conditional_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=64 \
  data=openwebtext-split \
  model=small \
  algo=mdlm \
  eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/openwebtext-train/mdlm_paper_ckpt/mdlm_ckpt.ckpt \
  sampling.num_sample_batches=1 \
  sampling.predictor=ancestral_cache \
  sampling.steps=128 \
  model.length=128 \
  +wandb.offline=true
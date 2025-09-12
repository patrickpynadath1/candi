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

CUDA_VISIBLE_DEVICCES=1 python main.py \
  mode=sample_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=1 \
  data=openwebtext-split \
  model=small \
  algo=duo_base \
  sampling.num_sample_batches=16 \
  sampling.steps=128 \
  eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/openwebtext-train/duo-paper-checkpoint-orig/duo.ckpt \
  +wandb.offline=true;
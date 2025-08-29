#!/bin/bash
#SBATCH -J sample_ar                  # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

steps=$2

export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=$1 python main.py \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=32 \
  data=lm1b-gpt2other \
  algo=duo_base \
  model=small \
  model.length=128 \
  eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/lm1b/2025.06.18/duo_baseline/duo_distill_lm1b_25000.ckpt \
  sampling.num_sample_batches=4 \
  sampling.steps=$steps \
  +wandb.offline=true \
  sampling.noise_removal=greedy

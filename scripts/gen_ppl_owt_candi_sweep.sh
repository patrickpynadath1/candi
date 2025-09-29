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





# CKPT=orig-10k

# export HYDRA_FULL_ERROR=1
# eval_batch_size=8
# number_sample_batches=4
# mixed_coeff=0.5
# CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py \
#   mode=sample_eval \
#   loader.batch_size=16 \
#   loader.eval_batch_size=8 \
#   data=openwebtext-split \
#   model=small \
#   algo=candi \
#   sampling.num_sample_batches=$number_sample_batches \
#   eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/openwebtext-train/$CKPT/checkpoints/0-10000.ckpt \
#   model.length=1024 \
#   algo.top_k=100 \
#   algo.step_size=1 \
#   algo.mixed_coeff=$mixed_coeff \
#   algo.min_percentile=.01 \
#   algo.max_percentile=.25 \
#   +wandb.offline=true;



CKPT=.05-.4-10k

export HYDRA_FULL_ERROR=1
eval_batch_size=8
number_sample_batches=4
mixed_coeff=0.5
CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py \
  mode=sample_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  model=small \
  algo=candi \
  sampling.num_sample_batches=$number_sample_batches \
  eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/openwebtext-train/$CKPT/checkpoints/0-10000.ckpt \
  model.length=1024 \
  algo.top_k=100 \
  algo.step_size=1 \
  algo.mixed_coeff=$mixed_coeff \
  algo.min_percentile=.05 \
  algo.max_percentile=.4 \
  +wandb.offline=true;


CKPT=.1-.4-10k

export HYDRA_FULL_ERROR=1
eval_batch_size=8
number_sample_batches=4
mixed_coeff=0.5
CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py \
  mode=sample_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  model=small \
  algo=candi \
  sampling.num_sample_batches=$number_sample_batches \
  eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/openwebtext-train/$CKPT/checkpoints/0-10000.ckpt \
  model.length=1024 \
  algo.top_k=100 \
  algo.step_size=1 \
  algo.mixed_coeff=$mixed_coeff \
  algo.min_percentile=.1 \
  algo.max_percentile=.4 \
  +wandb.offline=true;


CKPT=unweighted-10k 

export HYDRA_FULL_ERROR=1
eval_batch_size=8
number_sample_batches=4
mixed_coeff=0.5
CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py \
  mode=sample_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  model=small \
  algo=candi \
  sampling.num_sample_batches=$number_sample_batches \
  eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/openwebtext-train/$CKPT/checkpoints/0-10000.ckpt \
  model.length=1024 \
  algo.top_k=100 \
  algo.step_size=1 \
  algo.mixed_coeff=$mixed_coeff \
  algo.min_percentile=.1 \
  algo.max_percentile=.4 \
  +wandb.offline=true;




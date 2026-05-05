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

eval_batch_size=8
mixed_coeff=.5
CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py \
  mode=sample_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  model=small \
  algo=candi \
  sampling.num_sample_batches=16 \
  eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/openwebtext-train/candi-r-linear-orig-redo/checkpoints/last.ckpt  \
  model.length=1024 \
  algo.sampler=cached \
  algo.cont_schedule_type=linear \
  algo.cont_schedule_target=r \
  eval.generated_samples_path=/home/patrick/.cache/discrete_diffusion/owt-gen/candi-r-linear-orig-redo/generated_samples/samples.json \
  algo.max_percentile=.25 \
  algo.min_percentile=.01 \
  algo.mixed_coeff=$mixed_coeff \
  +wandb.offline=true;


  # distill-r3-bwd
  # eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/openwebtext-train/owt-1024-latest5/checkpoints/best.ckpt \

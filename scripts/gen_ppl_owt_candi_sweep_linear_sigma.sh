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
  eval.checkpoint_path=/scratch/gautschi/ppynadat/cont_diff/openwebtext-train/candi-linear-sigma-50k/checkpoints/last.ckpt \
  eval.generated_samples_path=$SCRATCH_DIR/cont_diff/owt-gen/candi-linear-sigma-.01-.25/gen_samples.json \
  eval.compute_eval_metrics=False \
  algo.cont_schedule_target=sigma \
  algo.cont_schedule_type=linear \
  algo.max_percentile=.25 \
  algo.min_percentile=.01 \
  model.length=1024 \
  algo.sampler=cached \
  algo.mixed_coeff=$mixed_coeff \
  +wandb.offline=true;


  # distill-r3-bwd
  # eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/openwebtext-train/owt-1024-latest5/checkpoints/best.ckpt \

#!/bin/bash

steps=$2
temp=$3
CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=candi \
        eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/text8/2025.09.19/054800/checkpoints/20-28500.ckpt \
        algo.temp=$temp \
        sampling.steps=$steps;


# steps=$2
# CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=candi \
#         eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/text8/2025.09.30/160147/checkpoints/last.ckpt \
#         sampling.steps=$steps;


# forward kl 
# /data/remote_cache/patrick/discrete_diffusion/text8/2025.09.30/225214/checkpoints/best.ckpt
# steps=$2
# CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=candi \
#         eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/text8/2025.09.30/225214/checkpoints/last.ckpt \
#         sampling.steps=$steps;

# mse on embeddings 
# /data/remote_cache/patrick/discrete_diffusion/text8/2025.09.30/230343/checkpoints/best.ckpt
# steps=$2
# CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=candi \
#         eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/text8/2025.09.30/230343/checkpoints/last.ckpt \
#         sampling.steps=$steps;


# gradual exponential increase 
# /data/remote_cache/patrick/discrete_diffusion/text8/2025.09.30/230343/checkpoints/best.ckpt
# steps=$2
# CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=candi \
#         eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/text8/2025.10.01/004302/checkpoints/last.ckpt \
#         sampling.steps=$steps;

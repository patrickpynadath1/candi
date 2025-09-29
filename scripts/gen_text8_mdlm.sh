#!/bin/bash
steps=$2
CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=mdlm \
        eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/text8/2025.09.19/055045/checkpoints/20-28500.ckpt \
        sampling.steps=$steps;
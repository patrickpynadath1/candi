#!/bin/bash
steps=$2
temp=$3
CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=candi \
        eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/text8/2025.10.16/220323/checkpoints/best.ckpt \
        algo.is_embed=True \
        algo.mixed_coeff=0 algo.min_percentile=.05 algo.max_percentile=.35\
        algo.temp=$temp \
        algo.dead_tokens=1000 \
        algo.pure_continuous=true sampling.steps=$steps;


#!/bin/bash
steps=$2
CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=config_text8 mode=sample_eval algo=candi \
        eval.checkpoint_path=/data/remote_cache/patrick/discrete_diffusion/text8/2025.09.22/182749/checkpoints/20-28500.ckpt \
        algo.mixed_coeff=0 algo.use_embedding_cache=False algo.min_percentile=.05 algo.max_percentile=.35\
        algo.pure_continuous=true sampling.steps=$steps;


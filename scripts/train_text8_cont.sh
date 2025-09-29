#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python main.py \
        algo=candi algo.mixed_coeff=0 algo.use_embedding_cache=False \
        algo.min_percentile=.05 algo.max_percentile=.35 trainer.val_check_interval=5000 algo.pure_continuous=true \
        checkpointing.resume_ckpt_path=/data/remote_cache/patrick/discrete_diffusion/text8/2025.09.22/130356/checkpoints/last.ckpt;
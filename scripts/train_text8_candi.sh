#!/bin/bash


CUDA_VISIBLE_DEVICES=$1 python main.py algo=candi_distill \
        algo.min_percentile=.05 \
        algo.max_percentile=.35 \
        wandb.name="distill_candi" \
        trainer.val_check_interval=4000 \
        training.finetune_path=/data/remote_cache/patrick/discrete_diffusion/text8/2025.09.19/054800/checkpoints/20-28500.ckpt;

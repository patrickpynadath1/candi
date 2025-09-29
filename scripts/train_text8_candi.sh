#!/bin/bash


MIN_PERCENTILE=$2
MAX_PERCENTILE=$3

CUDA_VISIBLE_DEVICES=$1 python main.py algo=candi \
        algo.min_percentile=$MIN_PERCENTILE \
        algo.max_percentile=$MAX_PERCENTILE \
        wandb.name="candi_${MIN_PERCENTILE}_${MAX_PERCENTILE}" \
        trainer.val_check_interval=5000;
#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python main.py \
        algo=candi algo.mixed_coeff=0  \
        algo.min_percentile=.05 algo.max_percentile=.35 trainer.val_check_interval=5000 algo.pure_continuous=true \
        algo.is_embed=True;
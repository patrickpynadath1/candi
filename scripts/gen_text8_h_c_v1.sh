#!/bin/bash

steps=$2

CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py --config-name=config_h_candi_v1_text8 mode=sample_eval  \
        eval.checkpoint_path="/data/remote_cache/patrick/discrete_diffusion/text8/2026.04.11/221309/checkpoints/last.ckpt" \
        sampling.steps=$steps;

#!/bin/bash

steps=$2

CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py --config-name=config_h_candi_v1_text8 mode=sample_eval algo=cosmos_ldm  \
        eval.checkpoint_path='${checkpoint_aliases.cosmos_ldm_no_self_cond}' \
        algo.encoder_decoder_path='${checkpoint_aliases.cosmos_ed_v2}' 
        algo.diffusion.use_self_cond=False \
        algo.latent_mode=encoder_decoder \
        sampling.steps=$steps;

#!/bin/bash

steps=$2
vocab=$3

CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py \
  --config-name=wiki_${vocab} \
  mode=sample_eval \
  eval.checkpoint_path="\${checkpoint_aliases.ced_ldm_uc_${vocab}}" \
  algo.encoder_decoder_path="\${checkpoint_aliases.ced_uc_${vocab}}" \
  algo.diffusion.use_self_cond=True \
  model.k_register_latents=128 \
  algo=cosmos_ldm \
  sampling.steps=$steps

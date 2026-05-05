#!/bin/bash

steps=$2
vocab=$3

CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py \
  --config-name=wiki_${vocab} \
  mode=sample_eval \
  eval.checkpoint_path="\${checkpoint_aliases.mdlm_${vocab}}" \
  algo=mdlm \
  sampling.steps=$steps

#!/bin/bash

steps=$2
vocab=$3

CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py \
  --config-name=wiki_${vocab} \
  mode=sample_eval \
  algo=cont_diff_onehot \
  eval.checkpoint_path="\${checkpoint_aliases.cd_oh_${vocab}}" \
  sampling.steps=$steps

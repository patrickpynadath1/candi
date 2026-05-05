#!/bin/bash

steps=$2
vocab=$3

CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py \
  --config-name=wiki_${vocab} \
  mode=sample_eval \
  algo=candi \
  algo.cont_schedule_type=linear \
  algo.cont_schedule_target=r \
  algo.max_percentile=.4 \
  algo.min_percentile=.01 \
  eval.checkpoint_path="\${checkpoint_aliases.candi_${vocab}}" \
  sampling.steps=$steps

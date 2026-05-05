#!/bin/bash

steps=$2

CUDA_VISIBLE_DEVICES=$1 python main_for_sweeps.py --config-name=text8 mode=sample_eval algo=ar  \
        checkpoint_aliases=text8-ref-models \
        data=text8_128 \
        eval.checkpoint_path='${checkpoint_aliases.ar-ref-128}' \
        model=small \
        sampling.steps=$steps;

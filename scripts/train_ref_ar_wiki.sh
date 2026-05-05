#!/bin/bash

for VOCAB in 512; do
  
  CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name=wiki_${VOCAB} \
    algo=ar \
    model=small \
    trainer.max_steps=20000 \
    sampling.steps=$steps
done

#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python main.py algo=mdlm trainer.val_check_interval=5000;
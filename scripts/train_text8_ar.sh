#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python main.py algo=ar trainer.val_check_interval=5000;
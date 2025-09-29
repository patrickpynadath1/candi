#!/bin/bash


export HYDRA_FULL_ERROR=1


number_sample_batches=10
file_name=candi_runtime_1024_orig.txt
touch $file_name

for steps in 8 16 32 64 128 256 512 1024
do 
    for temp in .9
    do  
        echo "Steps: $steps, Temp: $temp" >> $file_name

        mixed_coeff=0.5
        CUDA_VISIBLE_DEVICES=$1 python main.py \
        mode=sample_eval \
        loader.batch_size=16 \
        loader.eval_batch_size=1 \
        data=openwebtext-split \
        model=small \
        algo=candi \
        sampling.num_sample_batches=$number_sample_batches \
        sampling.steps=$steps \
        eval.checkpoint_path=/home/patrick/.cache/discrete_diffusion/openwebtext-train/owt-1024-latest5/checkpoints/best.ckpt \
        model.length=1024 \
        algo.top_k=100 \
        algo.step_size=1 \
        algo.mixed_coeff=$mixed_coeff \
        algo.temp=$temp \
        algo.use_embedding_cache=False \
        +wandb.offline=true >> $file_name;
    done
done
#!/bin/bash


export HYDRA_FULL_ERROR=1


file_name=cond_sweep_candi_prefix.txt
touch $file_name

for steps in 64 128
do
    for temp in .9 
    do
        echo "Steps: $steps, Temp: $temp" >> $file_name
        bash scripts/cond_eval_candi.sh $1 $steps $temp  >> $file_name
    done
done
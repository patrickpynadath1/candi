#!/bin/bash


export HYDRA_FULL_ERROR=1


file_name=cond_sweep_duo_suffix.txt
touch $file_name

for steps in 8 16 32 64 
do
    for temp in 1.0 .975 .95 .925 .9 .875 .85 .825 .8
    do
        echo "Steps: $steps, Temp: $temp" >> $file_name
        bash scripts/cond_eval_duo.sh $1 $steps $temp  >> $file_name
    done
done
#!/bin/bash


export HYDRA_FULL_ERROR=1


file_name=cond_sweep_ar_prefix.txt
touch $file_name


for temp in 1.0 .975 .95 .925 .9 .875 .85 .825 .8
do
    echo "Steps: $steps, Temp: $temp" >> $file_name
    bash scripts/cond_eval_ar.sh $1 $temp  >> $file_name
done
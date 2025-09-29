#!/bin/bash


export HYDRA_FULL_ERROR=1



number_sample_batches=128
touch output_entropy_ppl_sweep_ar.txt

for temp in .8 .825 .85 .875 .9 .925 .95 .975 1.0 1.05 1.1 1.15 1.2
do  
    echo $1
    echo " Temp: $temp" >> output_entropy_ppl_sweep_ar.txt
    bash scripts/gen_ppl_owt_ar.sh $1 $temp $number_sample_batches >> output_entropy_ppl_sweep_ar.txt
done
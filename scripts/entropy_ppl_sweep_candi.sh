#!/bin/bash
export HYDRA_FULL_ERROR=1
touch output_entropy_ppl_sweep_candi.txt
number_sample_batches=128

for steps in 1024 
do
    for temp in 1.0 .975 .95 .925 .9 .875 .85 .825 .8
    do  
        echo "Steps: $steps, Temp: $temp" >> output_entropy_ppl_sweep_candi.txt
        bash scripts/gen_ppl_owt_candi.sh $1 $steps $temp $number_sample_batches >> output_entropy_ppl_sweep_candi.txt
    done
done
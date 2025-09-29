#!/bin/bash
export HYDRA_FULL_ERROR=1

file_name=entropy_ppl_sweep_candi_hard_gradcalc1.txt
touch $file_name
number_sample_batches=128

for steps in 256 512
do
    for temp in .6 .625 .65 .675 .7 .725 .75 .775 .8 .825 .85 .875 .9 .925 .95 .975 1.0
    do  
        echo "Steps: $steps, Temp: $temp" >> $file_name
        bash scripts/gen_ppl_owt_candi.sh $1 $steps $temp $number_sample_batches >> $file_name
    done
done
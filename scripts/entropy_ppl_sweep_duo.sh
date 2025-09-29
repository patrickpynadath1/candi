#!/bin/bash


export HYDRA_FULL_ERROR=1



number_sample_batches=128
touch output_entropy_ppl_sweep_duo.txt

for steps in 128
do
    for temp in .6 .625 .65 .675 .7 .725 .75 .775
    do  
        echo "Steps: $steps, Temp: $temp" >> output_entropy_ppl_sweep_duo.txt
        bash scripts/gen_ppl_owt_duo1.sh $1 $steps $temp $number_sample_batches >> output_entropy_ppl_sweep_duo.txt
    done
done
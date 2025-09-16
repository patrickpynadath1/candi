#!/bin/bash


export HYDRA_FULL_ERROR=1


number_sample_batches=128
touch output_entropy_ppl_sweep_mdlm.txt

for steps in 1024
do 
    for temp in .8
    do  
        echo "Steps: $steps, Temp: $temp" >> output_entropy_ppl_sweep_mdlm.txt
        bash scripts/gen_ppl_owt_mdlm.sh $1 $steps $temp $number_sample_batches >> output_entropy_ppl_sweep_mdlm.txt
    done
done
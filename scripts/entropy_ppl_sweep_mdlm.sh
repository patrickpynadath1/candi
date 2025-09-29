#!/bin/bash


export HYDRA_FULL_ERROR=1


number_sample_batches=128
touch output_entropy_ppl_sweep_mdlm_lowtemp.txt

for steps in 512
do 
    for temp in .7 .725 .75 .775 
    do  
        echo "Steps: $steps, Temp: $temp" >> output_entropy_ppl_sweep_mdlm.txt
        bash scripts/gen_ppl_owt_mdlm.sh $1 $steps $temp $number_sample_batches >> output_entropy_ppl_sweep_mdlm.txt
    done
done
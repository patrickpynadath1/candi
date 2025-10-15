#!/bin/bash 
file_name=text8_cont_sweep_fixed_mask.txt
touch $file_name
for steps in 4 8 16 32 64 128
do 
    echo "Steps: $steps" >> $file_name
    bash scripts/gen_text8_cont_fixed_mask.sh $1 $steps >> $file_name
done
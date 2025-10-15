#!/bin/bash 
file_name=text8_mdlm_sweep.txt
touch $file_name
for steps in 4 8 16 32 64 128
do 
    for temp in  .1
    do
        echo "Steps: $steps, Temp: $temp" >> $file_name
        bash scripts/gen_text8_mdlm.sh $1 $steps $temp 
    done
done
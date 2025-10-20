#!/bin/bash 
file_name=text8_cont_sweep_fixed_mask.txt
touch $file_name
for steps in 4 8 16 32 64 128
do 
    for temp in .1 .2 .3 .4 .5 .6 .7 .8 .9 1.0
    do
        echo "Steps: $steps, Temp: $temp" >> $file_name
        bash scripts/gen_text8_cont_fixed_mask.sh $1 $steps $temp >> $file_name
    done
done
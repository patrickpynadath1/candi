#!/bin/bash 
file_name=text8_cont_corrupt_sweep.txt
touch $file_name
for steps in 8 16 32 64 128 256
do 
    for temp in .1 .2 .3 .4 .5 .6 .7 .8 .9 1.0 1.1 1.2
    do
        echo "Steps: $steps, Temp: $temp" >> $file_name
        bash scripts/gen_text8_cont_corrupted.sh $1 $steps $temp >> $file_name
    done
done
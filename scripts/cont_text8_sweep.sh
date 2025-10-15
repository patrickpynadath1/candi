#!/bin/bash 
file_name=text8_cont_sweep.txt
touch $file_name
for steps in 64
do 
    for temp in .01 .001
    do
        echo "Steps: $steps, Temp: $temp" >> $file_name
        bash scripts/gen_text8_cont.sh $1 $steps $temp 
    done
done
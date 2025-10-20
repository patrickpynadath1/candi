#!/bin/bash 
file_name=text8_candi_sweep2.txt
touch $file_name
for steps in 4 8 16 32 64 128
do 
    for temp in .01 .001
    do
        echo "Steps: $steps, Temp: $temp" >> $file_name
        bash scripts/gen_text8_candi_tmp.sh $1 $steps $temp >> $file_name
    done
done
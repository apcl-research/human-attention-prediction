#!/bin/bash

cd data/wallace_study
participant_list=("P3" "P4" "P5" "P6" "P7" "P8" "P9" "P10" "P11" "P12") 
for p in "${participant_list[@]}"; do
    rm -r participant$p
    cp -r  ./sample participant$p
    cd participant$p
    rm -r testset/
    mkdir testset/
    python3 testdatagen.py --holdout=$p
    cd ..
    sleep 5
 done

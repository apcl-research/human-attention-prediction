#!/bin/bash

cd data/smith_study
participant_list=( "p2"  "p4"  "p6" "p7" "p8" "p9" "p10" "p11"  "p13" "p14" "p15" "p16" "p17" "p18"  "p20" "p21") 
for p in "${participant_list[@]}"; do
    rm -r participant$p
    cp -r  ./sample participant$p
    cd participant$p
    rm -r testset/
    mkdir testset
    python3 testdatagen.py --holdout=$p
    cd ..
    sleep 5
 done


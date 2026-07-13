#!/bin/bash


cd data/rodeghero_study

participant_list=(KGT001 KGT002 KGT003 KGT004 KGT005 KGT007 KGT008 KGT009 KGT010) 
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

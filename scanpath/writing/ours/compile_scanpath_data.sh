#!/bin/bash
mkdir data/scanpath_data_5tokens

cd data/scanpath_data_5tokens

TOKEN=5

for file in /nfs/projects/scanpath/scanpath_prediction/bansal_dataset/writing/byFunction_*/; do

    id=$(echo "$file" | grep -oP 'byFunction_\K[0-9]+')
    mkdir $id
    cp ../scanpath/* ./$id
    cd $id
    rm test/*
    rm bins/*
    mkdir bins
    rm tmp/*
    mkdir tmp
    rm pkls/*
    mkdir pkls
    mkdir test
    python3 prepare_fc_raw.py --fundats-file=$file/eyeseq_train --num-tokens=$TOKEN
    python3 testdatagen.py --fundats-file=$file/eyeseq_test --num-tokens=$TOKEN
    rm pkls/*
    rm bins/*
    rm tmp/*
    cd ..

done


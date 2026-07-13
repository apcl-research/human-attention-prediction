# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
from datasets import Dataset

import pickle
import random
import argparse

random.seed(1337)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--data-file', type=str, default='/nfs/projects/rodeghero_study.pkl')
    parser.add_argument('--holdout', type=str, default='KGT001')
    parser.add_argument('--data-dir', type=str, default='bins/')
    

    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    data_file = args.data_file
    holdout = args.holdout
    data_dir = args.data_dir

    fixation_data = pickle.load(open(data_file, 'rb'))

    count = 0 
    duration_list = []  
    for key in fixation_data:
        participant = key
        if(participant != holdout):
            continue
        for data in fixation_data[key]:
            token = data["token"]
            ptgt = data["duration"]
            function = data["function"]
            next_tokens = '\t'.join(data["next_token"])
            with open(f'testset/{count}_{key}', 'w') as f:
                f.write(f'TDAT:\t{function}\nFIXATION:\t<s>{token}</s>\nNEXTTOKEN:\t<s>{next_tokens}</s>\nPTGT:\t{ptgt}' )
            count += 1
print(count)

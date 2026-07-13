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
import bincomb

from pathlib import Path

random.seed(1337)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--fundats-file', type=str, default='/nublar/eyeseq/reading/n10/byFunction_22624602/eyeseq_test/')
    parser.add_argument('--num-tokens', type=int, default=10)
    parser.add_argument('--data-dir', type=str, default='test/')

    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    fundats_file = args.fundats_file
    num_tokens = args.num_tokens
    data_dir = args.data_dir


    folder = Path(fundats_file)

    data = {}

    for file_path in folder.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            data[file_path.name] = f.read()

    fundats_fids = list(data.keys())

    
    count = 0 

    fundats_fids_2pt_px = fundats_fids[:]

    for fid in tqdm(fundats_fids_2pt_px):

        code = data[fid].split("SEQ:")[0]
        scanpath = data[fid].split("SEQ:")[-1]
        scanpath = scanpath.split("<s>")[-1]
        scanpath = scanpath.split("</s>")[0].strip()
        scanpath = scanpath.split(" ")[:num_tokens]
        try:
            with open(f'test/{fid}', 'w') as f:
                f.write(f'TDAT:\t{code}\nSEQ:\t' )
                count += 1
        except KeyError:
            continue
    print(count)

        
    
    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')

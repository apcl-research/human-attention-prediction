"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model_no_corr import GPTConfig, GPT
import re
import tqdm
from scipy.stats import pearsonr
# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 1 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
testdir = 'data/eyereward/testset/'
use_srcml = False
outfilename = "ckpt.pt"
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)



# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, outfilename)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)


arr = os.listdir(testdir)
def numeric_prefix(filename):
    match = re.match(r'(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')  # fallback

arr = sorted(arr, key=numeric_prefix)

testfiles = []
for file in arr:
    testfiles.append(testdir + file)

import math
import csv
#c = 0
human_data = []
attn_score_list = []
for testfile in tqdm.tqdm(testfiles[:]):
    with open(testfile, 'r') as f:
        start = f.read()
    
    match = re.search(r"^(.*?PTGT:\s*)([0-9.]+)", start, re.DOTALL)
    if match:
        start = match.group(1)
        ptgt = float(match.group(2))
        human_data.append(ptgt)
    start = start.split("NEXTTOKEN:")[0]
    source_code = start.split("FIXATION:\t")[0]
    source_code_len = len(encode(source_code))
    start_ids = encode(start)
    match = re.search(r"FIXATION:\s*<s>.*?</s>", start)
    if match:
        fixation = match.group(0)
        fixation_token_len = len(encode(fixation))
    else:
        fixation_token_len = 10
    
    
    if(len(start_ids) > 1024):
        start_ids = start_ids[0-1024:]

    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y, attn_score = model.generate(x, max_new_tokens, len(start_ids) - fixation_token_len, len(start_ids)-1, temperature=temperature, top_k=top_k)
                ret = decode(y[0].tolist())
                match = re.search(r'(NEXTTOKEN:.*)', ret)
                      
                attn_score = attn_score[0]
                attn_score_list.append(attn_score)
            if(len(attn_score_list) % 50 == 0 and len(attn_score_list) > 1):
                corr, _ = pearsonr(human_data, attn_score_list)
                print(corr)



human_and_predicted_outfile_name = os.path.join(out_dir, "result.pkl")
output_dict = {"human_data": human_data, "predicted_data":attn_score_list}
pickle.dump(output_dict, open(human_and_predicted_outfile_name, "wb"))
m = re.search(r"participant(P[0-9]+)", testdir)
holdout = m.group(1)
file_path = "correlation_rodeghero_study_no_corr.csv"
file_exists = os.path.isfile(file_path)
if(not file_exists):
     with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant", "pearson_correlation"])
with open(file_path, mode="a", newline="") as file:
    fieldnames = ["holdout", "pearson_corr"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    rows = [{"holdout": holdout, "pearson_corr": float(corr)}]
    writer.writerows(rows)

print(corr)

import os
from anthropic import Anthropic
from typing import List, Dict, Optional, Any
import argparse
import json
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
import re

key = ""
with open('/home/chiayi/claude.key', 'r') as f:
    key = f.readline().strip()


model_name = "gpt-5-nano"

def ask_gpt(precise_prompt):
  message = [{"role":"system", "content":"Read and understand source code."},
             {"role":"user", "content": precise_prompt}]

  completion = client.chat.completions.create(
        model=model_name,
        messages=message
  )
  ret = completion.choices[0].message.content
  return ret

def generate_scanpath(code):
  prompt = f'Suppose you are a programmer. Your task is to understand the source code. Given the source code {code}, can you tell me the first six important words in the scanpath that you would use to comprehend the provided source code. By important word, I mean the words that would help you understand the function. By scanpath, I mean the sequence of fixations that a person looks long enough to intae the information. Please use the template <s><token1>\t<token2>\t....\t<token6><\s> to provide your answer.'
  #len(prompt)
  answer = ask_gpt(prompt)
  return answer


client = Anthropic(
  # defaults to os.environ.get("ANTHROPIC_API_KEY")
  api_key=key
)








if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--fid', type=str)
    #parser.add_argument('--testdir', type=str, default='/nublar/eyeseq/reading/n10/byFunction_1046788/eyeseq_test')
    parser.add_argument('--prediction-filename', type=str, default='predict_1046788.txt')
    args = parser.parse_args()
    #testdir = args.testdir
    prediction_filename = args.prediction_filename
    fid = args.fid

    testdir = f"/nublar/eyeseq/reading/n10/byFunction_{fid}/eyeseq_test"

    testarr = os.listdir(testdir)
    testfiles = []
    pf = open(f'scanpath_predictions_gpt/{prediction_filename}', 'w')
    for file in testarr:
        if file.endswith('.txt'):
            testfiles.append(testdir +"/" + file)
    newdat = list()
    for file in tqdm(testfiles):
        with open(file, 'r') as f:
            input_seq = f.read()
            code_match = re.search(r"TDAT:(.*?)SEQ:", input_seq, re.DOTALL)
            seq_match = re.search(r"SEQ:(.*)", input_seq, re.DOTALL)

            code = code_match.group(1).strip() if code_match else ""
            answer = generate_scanpath(code)
            fid = file.split('/')[-1]
            fid = fid.split('.txt')[0]
            pf.write(f'{fid}\t{answer}\n')

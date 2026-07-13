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
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


key = ""
with open('/home/chiayi/claude.key', 'r') as f:
    key = f.readline().strip()


def build_prompt(code):

    ## 10222579
    ## 106_10222579.txt
    ## 111_10222579.txt
    ## 117_10222579.txt
    ## 129_10222579.txt
    ## 133_10222579.txt

    example1 = '''
        public boolean equals(Object obj) {
            if (obj == null || (obj.getClass() != getClass())) {
                return false;
            }
            CSSConditionalSelectorImpl s = (CSSConditionalSelectorImpl)obj;
            return (s.simpleSelector.equals(simpleSelector) &&
            s.condition.equals(condition));
        }
    '''
    example_scanpath1 = '''boolean obj.1 null CSSConditionalSelectorImpl =='''

    example_scanpath2 = '''public boolean false public boolean'''

    example_scanpath3 = '''Object boolean Object s.simpleSelector.equals false'''

    example_scanpath4 = '''Object equals CSSConditionalSelectorImpl getClass.1 Object'''

    example_scanpath5 = '''obj.2 equals boolean CSSConditionalSelectorImpl CSSConditionalSelectorImpl.1'''
    prompt = f'Here is the example source code {example1} and example five example scanpath that programmers use to understand this source code: {example_scanpath1}, {example_scanpath2}, {example_scanpath3}, {example_scanpath4}, and {example_scanpath5}. Suppose you are a programmer. Your task is to understand the source code. Given the source code {code}, can you tell me the first six important words in the scanpath that you would use to comprehend the provided source code. By important word, I mean the words that would help you understand the function. By scanpath, I mean the sequence of fixations that a person looks long enough to intae the information. Please use the template <s><token1>\t<token2>\t....\t<token5><\s> to provide your answer.'


    return prompt 



if __name__=='__main__':
    
    client = Anthropic(
        api_key=key
    )
    fid_list = [
        10222579, 14467418, 17121898, 18418213, 19498280, 19682824,
        26285656, 29601536, 33719869, 36405409, 39120328, 39299426,
        4114383, 45047585, 4627680, 49250848, 50994916, 12723449,
        14477536, 1782360, 18421665, 19505695, 20787007, 28631042,
        31696447, 34105249, 36634895, 39233258, 39840471, 41287529,
        45130358, 47570692, 49866815, 50995324, 12725774, 16777940,
        1810081, 19218425, 19507414, 23014476, 29318894, 31789275,
        34427273, 38221424, 39233866, 40865212, 43137003, 45147874,
        47571713, 50026101, 51577053, 13482891, 1694531, 18354735,
        19413001, 19507735, 26215341, 29572299, 33718481, 34604973,
        38737384, 39298970, 40936756, 44521080, 46026508, 47571922,
        50891793, 7957602
    ]
    prompt_list = {}


    for fid in tqdm(fid_list):
        testdir = f"/nfs/projects/scanpath/scanpath_prediction/bansal_dataset/writing/byFunction_{fid}/eyeseq_test"

        testarr = os.listdir(testdir)
        testfiles = []
        for file in testarr:
            if file.endswith('.txt'):
                testfiles.append(testdir +"/" + file)
        newdat = list()
        for file in testfiles:
            with open(file, 'r') as f:
                input_seq = f.read()
                code_match = re.search(r"TDAT:(.*?)SEQ:", input_seq, re.DOTALL)
                seq_match = re.search(r"SEQ:(.*)", input_seq, re.DOTALL)

                code = code_match.group(1).strip() if code_match else ""

                #answer = generate_scanpath(code)
                function_fid = file.split('/')[-1]
                funcion_fid = function_fid.split('.txt')[0]
                prompt_list[funcion_fid] = build_prompt(code)
    requests = []
    for fid in prompt_list:
        requests.append(
            Request(
                custom_id=fid,
                params=MessageCreateParamsNonStreaming(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_list[fid],
                        }
                    ],
                ),
            ))

    message_batch = client.messages.batches.create(requests = requests)
    print(message_batch)


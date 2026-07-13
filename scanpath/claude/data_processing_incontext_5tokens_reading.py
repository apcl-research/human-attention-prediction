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

    example1 = '''
     public String capitalizeString(String s) {
        String result = "";

        for(int i = 0; i < s.length(); i++) {
            if(i == 0 || s.substring(i-1, i).equals(" "))
                result += s.substring(i, i + 1).toUpperCase();
            else
                result += s.substring(i, i + 1);

        }

        return result;
    }
    '''
    example_scanpath1 = '''capitalizeString String String.2 0 < 0'''

    example_scanpath2 = '''= s.substring result capitalizeString String.1'''

    example_scanpath3 = '''int i =.1 i.1 int'''

    example_scanpath4 = '''s.length int s.substring.1 0.1 if'''

    example_scanpath5 = '''iresult String public String i'''

    prompt = f'Here is the example source code {example1} and example five example scanpath that programmers use to understand this source code: {example_scanpath1}, {example_scanpath2}, {example_scanpath3}, {example_scanpath4}, and {example_scanpath5}. Suppose you are a programmer. Your task is to understand the source code. Given the source code {code}, can you tell me the first six important words in the scanpath that you would use to comprehend the provided source code. By important word, I mean the words that would help you understand the function. By scanpath, I mean the sequence of fixations that a person looks long enough to intae the information. Please use the template <s><token1>\t<token2>\t....\t<token5><\s> to provide your answer.'


    return prompt 



if __name__=='__main__':
    
    
    client = Anthropic(
        api_key=key
    )
    fid_list = [
        1046788, 1118165, 11759898, 11950130, 1412807, 15689897, 16958722,
        1736289, 1810886, 18123253, 18251847, 18252350, 18420500, 18912425,
        18929060, 19280843, 19282261, 19344442, 19344536, 19346491, 19498298,
        20687719, 20950900, 21359951, 22407318, 22618655, 22622479, 22624602,
        22628734, 22907997, 24245709, 250694, 26412118, 26493872, 26501411,
        27798254, 27801498, 27802185, 27907979, 28953715, 2896279, 29852582,
        29854794, 29859244, 31203037, 31788771, 33519720, 33719114, 33719117,
        33719118, 33719607, 34413723, 34413807, 34413808, 34425716, 34426334,
        34426756, 34426938, 3456415, 3457090, 34609355, 35061399, 35553511,
        35553791, 37762493, 38184555, 38221537, 39215677, 3934822, 40099556,
        40776207, 40778768, 40865383, 40875567, 40879350, 4280405, 43040209,
        43040436, 43303607, 43419611, 4453291, 45888514, 45929468, 47479282,
        48104729, 48861766, 49121415, 51019251, 51122387
    ]

    prompt_list = {}


    for fid in tqdm(fid_list):
        testdir = f"/nfs/projects/scanpath/scanpath_prediction/bansal_dataset/reading/byFunction_{fid}/eyeseq_test"

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





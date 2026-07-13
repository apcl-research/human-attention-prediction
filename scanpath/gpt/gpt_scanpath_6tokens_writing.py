import pickle
import openai
from tqdm import tqdm
import argparse
import os
import re


key = ""
with open('/home/chiayi/openai.key', 'r') as f:
    key = f.readline().strip()

client = openai.OpenAI(api_key=key)

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

def generate_openai_sum(code):
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
    example_scanpath1 = '''boolean obj.1 null CSSConditionalSelectorImpl == equals'''
    
    example_scanpath2 = '''public boolean false public boolean =='''

    example_scanpath3 = '''Object boolean Object s.simpleSelector.equals false return'''

    example_scanpath4 = '''Object equals CSSConditionalSelectorImpl getClass.1 Object =='''

    example_scanpath5 = '''obj.2 equals boolean CSSConditionalSelectorImpl CSSConditionalSelectorImpl.1 boolean'''
    
    prompt = f'Here is the example source code {example1} and example five example scanpath that programmers use to understand this source code: {example_scanpath1}, {example_scanpath2}, {example_scanpath3}, {example_scanpath4}, and {example_scanpath5}. Suppose you are a programmer. Your task is to understand the source code. Given the source code {code}, can you tell me the first 10 important words in the scanpath that you would use to comprehend the provided source code. By important word, I mean the words that would help you understand the function. By scanpath, I mean the sequence of fixations that a person looks long enough to intae the information. Please use the template <s><token1>\t<token2>\t....\t<token6><\s> to provide your answer.'
  #len(prompt)
    answer = ask_gpt(prompt)
    return answer

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
    
    testdir = f"/nublar/eyeseq/writing/n10/byFunction_{fid}/eyeseq_test"

    testarr = os.listdir(testdir)
    testfiles = []
    pf = open(f'scanpath_predictions_6tokens_writing/{prediction_filename}', 'w')
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
            answer = generate_openai_sum(code)
            fid = file.split('/')[-1]
            fid = fid.split('.txt')[0]
            pf.write(f'{fid}\t{answer}\n')
        #seq = seq_match.group(1).strip() if seq_match else ""
        #seq = seq.split(" ")
        #seq.pop(0)   # remove first element
        #seq.pop(-1)  # remove last element
        #fid = file.split('/')[-1]
        #fid = fid.split('.txt')[0]

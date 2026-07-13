import argparse
import os
import editdistance as lev
import re
import pickle
from thefuzz import fuzz
import math


def fil(com):
    sremove = com.split(">",1)[-1] # remove start tag
    eremove = sremove.rsplit("<",1)[0] #remove end tag
    ret= eremove.strip()
    return ret

def getIndex(words,fid, N):
    fid = int(fid.split("_")[1])
    words = words.split(" ")
    words = words[:N]
    indexmap = pickle.load(open("indexwords.pkl","rb"))
    indexlist = indexmap[fid]
    indices = list()

    for word in words:
        try:
            index = indexlist[word]
        except:
            best = -1
            for iword in indexlist:
                score = fuzz.ratio(iword,word)
                if score> best:
                    best = score
                    index = indexlist[iword] # best match index if index not found exactly

        indices.append(chr(65+index))
    
    return "".join(indices)


def convert_to_relevances(reference, prediction):
    """
    Convert reference and prediction strings into
    NDCG-compatible relevance lists.

    Repetition allowed, but order-sensitive:
    earlier occurrences in reference have higher relevance.
    """
    n = len(reference)

    # Step 1: assign descending relevance
    rel_values = list(range(n, 0, -1))
    ref_pool = list(zip(reference, rel_values))
    used = [False] * n
    pred_rels = []

    # Step 2: match prediction to highest remaining relevance
    for ch in prediction:
        best_idx = -1
        best_rel = -1

        for i, (ref_ch, rel) in enumerate(ref_pool):
            if not used[i] and ch == ref_ch:
                if rel > best_rel:
                    best_rel = rel
                    best_idx = i

        if best_idx != -1:
            pred_rels.append(best_rel)
            used[best_idx] = True
        else:
            pred_rels.append(0)

    # Step 3: pad or truncate
    if len(pred_rels) < n:
        pred_rels += [0] * (n - len(pred_rels))
    else:
        pred_rels = pred_rels[:n]

    ideal_rels = rel_values
    return ideal_rels, pred_rels
def dcg(relevances, k=None):
    if k:
        relevances = relevances[:k]
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

def ndcg(ideal_relevances, pred_relevances, k=None):
    dcg_val = dcg(pred_relevances, k)
    idcg_val = dcg(ideal_relevances, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def compute_ratio(preds, refs):
    ref_fids = list(refs.keys())
    ref_vals = list(refs.values())
    total_score = 0
    
    for i in range(len(ref_fids)):
        ref_fid = ref_fids[i]
        ref_val = ref_vals[i]
        total_fun = len(ref_fids)
        pred_val = preds[ref_fid]
        
        score = fuzz.ratio(pred_val, ref_val)
        total_score += score
    total_score /= total_fun
    return total_score

def compute_ratiolist(preds, refs):
    ref_fids = list(refs.keys())
    ref_vals = list(refs.values())
    total_score = list()

    for i in range(len(ref_fids)):
        ref_fid = ref_fids[i]
        ref_val = ref_vals[i]
        pred_val = preds[ref_fid]
        maxnum = max(len(ref_val),len(pred_val))
        score = lev.eval(ref_val,pred_val)
        total_score.append(score/maxnum) # normalize for sequence length 10
    return total_score

def compute_ndcg(preds, refs):
    #print(refs)
    #print(preds)
    ref_fids = list(refs.keys())
    ref_vals = list(refs.values())
    total_score = list()

    for i in range(len(ref_fids)):
        ref_fid = ref_fids[i]
        ref_val = ref_vals[i]
        pred_val = preds[ref_fid]
        ideal_rels, pred_rels = convert_to_relevances(ref_val, pred_val)
        score = ndcg(ideal_rels, pred_rels)
        #maxnum = max(len(ref_val),len(pred_val))
        #score = lev.eval(ref_val,pred_val)
        total_score.append(score) # normalize for sequence length 10
    return total_score

def precision_at_k_no_reuse(reference, prediction, k):
    ref_pool = list(reference)
    used = [False] * len(ref_pool)
    
    relevant = 0
    for ch in prediction[:k]:
        for i, ref_ch in enumerate(ref_pool):
            if not used[i] and ch == ref_ch:
                relevant += 1
                used[i] = True
                break
    
    return relevant / k

def compute_precision_at_k(preds, refs, k):
    ref_fids = list(refs.keys())
    ref_vals = list(refs.values())
    total_score = list()

    for i in range(len(ref_fids)):
        ref_fid = ref_fids[i]
        ref_val = ref_vals[i]
        pred_val = preds[ref_fid]
        score = precision_at_k_no_reuse(ref_val, pred_val, k)
        total_score.append(score) # normalize for sequence length 10
    return total_score

def recall_at_k_no_reuse(reference, prediction, k):
    ref_pool = list(reference)
    used = [False] * len(ref_pool)
    relevant = 0

    for ch in prediction[:k]:
        for i, ref_ch in enumerate(ref_pool):
            if not used[i] and ch == ref_ch:
                relevant += 1
                used[i] = True
                break

    return relevant / len(ref_pool)


def compute_recall_at_k(preds, refs, k):
    ref_fids = list(refs.keys())
    ref_vals = list(refs.values())
    total_score = list()

    for i in range(len(ref_fids)):
        ref_fid = ref_fids[i]
        ref_val = ref_vals[i]
        pred_val = preds[ref_fid]
        score = recall_at_k_no_reuse(ref_val, pred_val, k)
        total_score.append(score) # normalize for sequence length 10
    return total_score



def main(input_file, reffilename,typeout, N):
    

    
    if input_file is None:
        print('Please provide an input file to test')
        exit()    

    preds = dict()
    predicts = open(input_file, 'r')

    n_all_fun = 0
    for c, line in enumerate(predicts):
        n_all_fun += 1
        (fid, pred) = line.split('\t')[0], line.split('\t')[-1]
        pred = fil(pred)
        if pred == 'none': # skip over fids where prediction is none/bugged.
            continue
        try:
            pred = getIndex(pred,fid, N)
        except:
            continue
        preds[fid] = pred
    predicts.close()

    refs = dict()
    targets = open(reffilename, 'r')

    for line in targets:
        (fid, com) = line.split('\t')[0], line.split('\t')[1]
        if fid not in preds.keys(): # skips refs we don't have preds for
            continue

        com = fil(com)
        if com.isspace():  #skip over fids where refernce is just whitespace
            continue
        com = getIndex(com,fid, N)
        refs[fid] = com


    if typeout == 'mean':
        ratio = round(compute_ratio(preds, refs)/100, 2) 
    elif typeout == 'list':
        ratio = compute_ratiolist(preds,refs)
        ndcg = compute_ndcg(preds,refs)
        precision_at_k = compute_precision_at_k(preds, refs, N)
        recall_at_k = compute_recall_at_k(preds, refs, N)
    
    return ratio, ndcg, precision_at_k, recall_at_k
    


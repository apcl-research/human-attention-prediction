import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import argparse
import csv
import os



# =========================================================
# 1. srcML Parsing: Code → AST Nodes + Sparse Adjacency
# =========================================================
def code_to_srcml(code: str, language: str = "java") -> str:
    """
    Convert source code into srcML XML string.
    Requires srcml to be installed (https://www.srcml.org).
    """
    process = subprocess.Popen(
        ["srcml", "--language", language],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True
    )
    xml_output, error = process.communicate(code)
    if process.returncode != 0:
        raise RuntimeError(f"srcML failed: {error}")
    return xml_output

def dfs_traverse(node, node_list, token_list, edges, parent_index=None):
    idx = len(node_list)
    node_list.append(node.tag)

    token = ""
    if node.tag == "literal":
        # Only keep actual literal content
        literal_type = node.attrib.get("type")
        if literal_type == "string":
            token = "<STR>"
        else:
            token = node.text.strip() if node.text else ""
    elif node.tag == "name" or node.tag == "operator":
        token = node.text.strip() if node.text else ""

    token_list.append(token)
    if parent_index is not None:
        edges.append((parent_index, idx))

    for child in node:
        dfs_traverse(child, node_list, token_list, edges, parent_index=idx)



def ast_to_sequence_and_matrix(xml_str: str):
    """
    Parse XML AST string and generate node sequence + adjacency sparse matrix.
    """
    root = ET.fromstring(xml_str)
    node_list, token_list, edges = [], [], []
    dfs_traverse(root, node_list, token_list, edges)

    n = len(node_list)
    row_idx, col_idx = [], []
    for src, dst in edges:
        row_idx.extend([src, dst])  # undirected edges
        col_idx.extend([dst, src])
    data = np.ones(len(row_idx), dtype=np.int8)
    adj_matrix = csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
    return node_list, token_list, adj_matrix



NAMESPACE = "http://www.srcML.org/srcML/src"

def strip_namespace(tag: str) -> str:
    prefix = "{" + NAMESPACE + "}"
    if tag.startswith(prefix):   # only remove exact srcML namespace
        return tag[len(prefix):]
    return tag   # leave untouched otherwise



def build_dataset_from_code(raw_dataset, vocab):
    for data in tqdm(raw_dataset[:]):
        token = data["token"]
        ptgt = data["duration"]
        function = data["function"]
        xml_str = data["srcml"]
        nodes, token_list, adj = ast_to_sequence_and_matrix(xml_str)
        tmpvocabs = ["UNK"] + nodes + token_list
        allvocabs = [strip_namespace(tag) for tag in tmpvocabs]
        x_indices = []
        for tag in allvocabs:
            if(tag in vocab):
                vocab[tag] += 1
            else:
                vocab.setdefault(tag, 1)
            if(token in vocab):
                vocab[token] += 1
            else:
                vocab.setdefault(token, 1)

    return vocab




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixation_data_path",
        type=str,
        required=True,
        help="Path to fixation data pickle file"
    )
    parser.add_argument(
        "--output_vocab_path",
        type=str,
        required=True,
        help="Path to dump vocab pickle file"
    )
    args = parser.parse_args()

    fixation_data = pickle.load(open(args.fixation_data_path, "rb"))
    raw_data = []
    vocab = {}

    for key in list(fixation_data.keys())[:]:
        alldata = fixation_data[key]
        for data in alldata:
            raw_data.append(data)
    vocab_count = build_dataset_from_code(raw_data, vocab)

    count = 0
    vocab_list = {}
    for vocab in vocab_count:
        vocab_list[vocab] = count
        count += 1
    pickle.dump(vocab_list, open(args.output_vocab_path, "wb"))



if __name__ == "__main__":
    main()


















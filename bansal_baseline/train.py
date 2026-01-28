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

nltk.download('punkt_tab')  # Download the tokenizer models

# =========================================================
# 1. srcML Parsing: Code → AST Nodes + Sparse Adjacency
# =========================================================
def code_to_srcml(code: str, language: str = "Java") -> str:
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



class ASTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(ASTModel, self).__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # GCN layers
        self.gnn1 = GCNConv(embed_dim, hidden_dim)
        self.gnn2 = GCNConv(hidden_dim, hidden_dim)

        # GRU after GCN
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        # Fully connected layers
        # fc1 takes GRU output + focal_points (embed_dim assumed for focal_points)
        self.fc1 = nn.Linear(hidden_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch_idx, focal_points):

        # Step 1: Node embeddings
        x = self.embedding(x)  # [total_nodes, embed_dim]

        # Step 2: GCN layers
        node_embeddings = self.gnn1(x, edge_index)
        node_embeddings = torch.relu(node_embeddings)
        node_embeddings = self.gnn2(node_embeddings, edge_index)
        node_embeddings = torch.relu(node_embeddings)

        # Step 3: Group node embeddings per graph
        batch_size = batch_idx.max().item() + 1
        node_sequences = []
        for i in range(batch_size):
            node_seq = node_embeddings[batch_idx == i]  # nodes of graph i
            node_sequences.append(node_seq)

        # Step 4: Pad sequences to the same length
        seq_lengths = [seq.size(0) for seq in node_sequences]
        max_len = max(seq_lengths)
        padded_sequences = []
        for seq in node_sequences:
            if seq.size(0) < max_len:
                pad = torch.zeros(max_len - seq.size(0), seq.size(1), device=seq.device)
                seq = torch.cat([seq, pad], dim=0)
            padded_sequences.append(seq)

        # Step 5: Stack sequences for GRU
        node_embeddings_seq = torch.stack(padded_sequences, dim=0).contiguous()  # [batch_size, max_len, hidden_dim]

        # Step 6: GRU
        gru_out, _ = self.gru(node_embeddings_seq)  # [batch_size, max_len, hidden_dim]

        # Step 7: Aggregate node features (mean over sequence)
        graph_embeddings = gru_out.mean(dim=1)  # [batch_size, hidden_dim]

        # Step 8: Concatenate focal points
        graph_embeddings = torch.cat([graph_embeddings, focal_points], dim=1)  # [batch_size, hidden_dim + embed_dim]

        # Step 9: Fully connected layers
        out = torch.relu(self.fc1(graph_embeddings))
        out = self.fc2(out)  # [batch_size, 1]
        out = torch.sigmoid(out)

        return out




# =========================================================
# 3. Utilities
# =========================================================
def scipy_to_edge_index(adj: csr_matrix):
    adj_coo = adj.tocoo()
    row = torch.tensor(adj_coo.row, dtype=torch.long)
    col = torch.tensor(adj_coo.col, dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


NAMESPACE = "http://www.srcML.org/srcML/src"

def strip_namespace(tag: str) -> str:
    prefix = "{" + NAMESPACE + "}"
    if tag.startswith(prefix):   # only remove exact srcML namespace
        return tag[len(prefix):]
    return tag   # leave untouched otherwise


def build_dataset_from_code(raw_dataset, vocab, embed_dim, unique_tokens, training=True):
    """
    Build PyG dataset from raw code.

    Args:
        raw_dataset: list of (function_name, function, token, duration, ptgt)
        vocab: dictionary mapping token/tag -> index
        embed_dim: embedding size (for focal point, if used)
        unique_tokens: extra tokens to include
        training: bool, if True allow vocab to grow, if False map unknown tokens to 0

    Returns:
        list of PyG Data objects
    """
    dataset = []
    for data in tqdm(raw_dataset[:]):
        token = data["token"]
        ptgt = data["duration"]
        function = data["function"]
        xml_str = data["srcml"]
        nodes, token_list, adj = ast_to_sequence_and_matrix(xml_str)
        edge_index = scipy_to_edge_index(adj)
        tmpvocabs = ["UNK"] + nodes + token_list
        allvocabs = [strip_namespace(tag) for tag in tmpvocabs]
        x_indices = []
        for tag in allvocabs:
            idx = vocab.get(tag, 0)
            x_indices.append(idx)
        x = torch.tensor(x_indices, dtype=torch.long)

        # Handle focal_point (task-specific feature)
        focal_indices = vocab.get(token, 0)

        # Target value
        y = torch.tensor([ptgt], dtype=torch.float)

        # Build PyG Data object
        graph = Data(x=x, edge_index=edge_index, y=y)
        graph.focal_point = focal_indices
        dataset.append(graph)
    return dataset







def train_model(raw_dataset, unique_tokens, vocab):

    embed_dim = 100
    hidden_dim = 100
    num_classes = 2
    dataset = build_dataset_from_code(raw_dataset, vocab, embed_dim, unique_tokens)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASTModel(vocab_size=len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(70)):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            focal_points = torch.cat([g.focal_point for g in batch.to_data_list()], dim=0).to(device).contiguous()
            if focal_points.dim() == 1:
                focal_points = focal_points.unsqueeze(1)
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            batch_idx = batch.batch.to(device)
            y = batch.y.float().to(device)
            out = model(x, edge_index, batch_idx, focal_points)
            loss = criterion(out.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss={total_loss:.4f}")
    return model

def predict(model, raw_dataset, vocab, unique_tokens):
    embed_dim = 100 # one participant dim = 80
    hidden_dim = 100
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = build_dataset_from_code(raw_dataset, vocab, embed_dim, unique_tokens, training=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    predictions = [] 
    groundtruth = []
    with torch.no_grad():
        for batch in loader:
            focal_points = torch.cat([g.focal_point for g in batch.to_data_list()], dim=0).to(device).contiguous()
            if focal_points.dim() == 1:
                focal_points = focal_points.unsqueeze(1)

            # Move data to device
            x, edge_index, batch_idx = batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device)
            y = batch.y
            groundtruth.extend(y.tolist())
            # Forward pass
            out = model(x, edge_index, batch_idx, focal_points)
            val = out.squeeze()  # remove extra dimensions
            # Convert to list and append
            if val.dim() == 0:  # scalar
                predictions.append(val.item())
            else:  # batch of predictions
                predictions.extend(val.tolist())

    return groundtruth, predictions






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of argparse")

    parser.add_argument("--holdout", type=str, default="P8")
    

    parser.add_argument(
        "--fixation_path",
        type=str,
        default="/nfs/projects/wallace_study.pkl",
        help="Path to fixation data pickle file"
    )

    parser.add_argument(
        "--vocab_path",
        type=str,
        default="/nfs/projects/wallace_study.pkl",
        help="Path to vocabulary pickle file"
    )


    args = parser.parse_args()

    holdout_set = args.holdout
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fixation_data = pickle.load(open(args.fixation_path, "rb"))
    vocab = pickle.load(open(args.vocab_path, "rb"))


    pearson_correlation = {}
    train_raw = []
    test_raw = []
    train_to_duration_map = {}

    unique_tokens = []
    for key in list(fixation_data.keys())[:]:
        participant = key.split("-")[0]
        alldata = fixation_data[key]
        if(participant == holdout_set):
            for data in alldata[:]:
                test_raw.append(data)
        else:
            for data in alldata[:]:
                train_raw.append(data)

    fixation_filename = os.path.basename(args.fixation_path)
    dataset_name = fixation_filename.replace(".pkl", "").split("_")[-2:]
    dataset_name = "_".join(dataset_name)
    # ---- output directories ----
    base_result_dir = f"results/{dataset_name}/{holdout_set}"
    os.makedirs(base_result_dir, exist_ok=True)

    # ---- train & save model ----
    model = train_model(train_raw, unique_tokens, vocab)
    torch.save(
        model.state_dict(),
        os.path.join(base_result_dir, f"model_weights_{holdout_set}.pth")
    )

    # ---- evaluation ----
    model.eval()
    groundtruth, predictions = predict(model, test_raw, vocab, unique_tokens)

    output_dict = {
        "human_data": groundtruth,
        "predicted_data": predictions
    }

    with open(
        os.path.join(base_result_dir, f"results_{holdout_set}.pkl"), "wb"
    ) as f:
        pickle.dump(output_dict, f)

    # ---- correlation ----
    corr, p_value = pearsonr(predictions, groundtruth)
    corr = corr.item()
    pearson_correlation[holdout_set] = corr

    # ---- CSV logging ----
    csv_path = f"results/correlation_{dataset_name}.csv"
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as file:
        fieldnames = ["holdout", "pearson_corr", "vocab_size"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "holdout": holdout_set,
            "pearson_corr": float(corr),
            "vocab_size": len(vocab)
        })




















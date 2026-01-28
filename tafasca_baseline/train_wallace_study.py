import torch
import pickle
import torch.nn as nn
import tiktoken
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
import os
import csv

class CodeDataset(Dataset):
    def __init__(self, code_list, cond_list, targets, tokenizer):
        self.code_list = code_list
        self.cond_list = cond_list
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.src_tokens = self.tokenizer.encode_batch(code_list)
        self.cond_tokens = self.tokenizer.encode_multi(cond_list)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.src_tokens[idx], self.cond_tokens[idx], self.targets[idx]


class CodeConditionedModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        enc_layers=4,
        dec_layers=2,
        mlp_hidden=128,
        max_len=1024,
    ):
        super().__init__()

        # --- Embeddings ---
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        # --- Encoder ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers)

        # --- MLP for conditioning token ---
        self.token_mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, d_model),
        )

        # --- Decoder ---
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)

        # --- Output ---
        self.output_head = nn.Linear(d_model, 1)
        self.max_len = max_len

    def forward(
        self,
        src_tokens,        # (B, L) source code token indices
        cond_tokens,       # (B, Lc) conditioning token indices (can be multi-token)
        src_padding_mask=None,
        cond_padding_mask=None,
    ):
        B, L = src_tokens.shape
        device = src_tokens.device

        # --- Source embeddings + positional ---
        pos_ids = torch.arange(L, device=device).unsqueeze(0)
        src = self.token_embed(src_tokens) + self.pos_embed(pos_ids)

        # --- Encode source code ---
        memory = self.encoder(src, src_key_padding_mask=src_padding_mask)

        # --- Conditioning token embeddings ---
        cond_embed = self.token_embed(cond_tokens)  # (B, Lc, D)
        if cond_padding_mask is not None:
            # Masked mean pooling for multi-token conditioning
            cond_embed = (cond_embed * (~cond_padding_mask.unsqueeze(-1)).float()).sum(1)
            cond_embed = cond_embed / (~cond_padding_mask).sum(1, keepdim=True).clamp(min=1)
        else:
            # Simple mean if no mask
            cond_embed = cond_embed.mean(1)  # (B,D)

        cond = self.token_mlp(cond_embed).unsqueeze(1)  # (B,1,D)

        # --- Decode ---
        dec_out = self.decoder(
            tgt=cond,
            memory=memory,
            memory_key_padding_mask=src_padding_mask,
        )

        # --- Output ---
        return torch.sigmoid(self.output_head(dec_out[:, 0]))  # (B,1)


class Tokenizer:
    def __init__(self, model_name="gpt2", max_len=1024):
        self.enc = tiktoken.get_encoding(model_name)
        self.max_len = max_len

    def encode_batch(self, texts):
        """
        texts: List[str]
        Returns: tensor (B, L)
        """
        tokens_list = [self.enc.encode(t)[: self.max_len] for t in texts]
        max_len = max(len(t) for t in tokens_list)
        max_len = min(max_len, self.max_len)

        batch = []
        for t in tokens_list:
            padded = t + [0] * (max_len - len(t))
            batch.append(padded)
        return torch.tensor(batch, dtype=torch.long)

    def decode(self, token_ids):
        return self.enc.decode(token_ids)
    def encode_multi(self, texts):
        """
        texts: list of strings (conditioning tokens)
        Returns: tensor (B, Lc) with all tiktoken tokens, padded
        """
        tokens_list = [self.enc.encode(t) for t in texts]
        max_len = max(len(t) for t in tokens_list)
        batch = []
        for t in tokens_list:
            padded = t + [0] * (max_len - len(t))
            batch.append(padded)
        return torch.tensor(batch, dtype=torch.long)


def predict_with_dataloader(model, dataloader, device="cpu"):
    model.eval()
    preds = []
    refs = []

    with torch.no_grad():
        for src, cond, targets in dataloader:
            refs.extend(targets.numpy())
            src = src.to(device)
            cond = cond.to(device)

            out = model(src, cond).squeeze(-1)
            preds.append(out.cpu())

    return refs, torch.cat(preds, dim=0).numpy()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of argparse")

    parser.add_argument("--holdout", type=str, default="P8")
    args = parser.parse_args()
    holdout_set = args.holdout
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    fixation_data = pickle.load(open("/nfs/projects/wallace_study.pkl", "rb"))
    train_raw = []
    test_raw = []

    for key in list(fixation_data.keys())[:]:
        participant = key.split("-")[0]
        alldata = fixation_data[key]
        if(participant == holdout_set):
            for data in alldata[:]:
                test_raw.append(data)
        else:
            for data in alldata[:]:
                train_raw.append(data)
    
    code_snippets = [item["function"] for item in train_raw]
    cond_texts = [item["token"] for item in train_raw]
    targets = [item["duration"] for item in train_raw]



    tokenizer = Tokenizer(max_len=1024)
    vocab_size = tokenizer.enc.n_vocab + 1  # correct vocab size
    model = CodeConditionedModel(vocab_size=vocab_size, d_model=128).to(device)

    dataset = CodeDataset(code_snippets, cond_texts, targets, tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    criterion = nn.MSELoss()  # regression target

    # Training loop
    model.train()
    for epoch in tqdm(range(10)):
        total_loss = 0
        for src_tokens, cond_tokens, target in dataloader:
            src_tokens = src_tokens.to(device)
            cond_tokens = cond_tokens.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(src_tokens, cond_tokens)  # (B,1)
            loss = criterion(output.squeeze(-1), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), f"./wallace_study/{holdout_set}/model_weights_{holdout_set}.pth")
    
    
    code_snippets = [item["function"] for item in test_raw]
    cond_texts = [item["token"] for item in test_raw]
    targets = [item["duration"] for item in test_raw]
    infer_dataset = CodeDataset(code_snippets, cond_texts, targets, tokenizer)
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=8,
        shuffle=False
    )

    refs, preds = predict_with_dataloader(model, infer_loader, device=device)
    output_dict = {"human_data": refs, "predicted_data":preds}
    pickle.dump(output_dict, open(f"./wallace_study/{holdout_set}/results_{holdout_set}.pkl", "wb"))
    r, p = pearsonr(refs, preds)
    file_path = "./wallace_study/correlation_wallace_study.csv"
    file_exists = os.path.exists(file_path)

    with open(file_path, mode="a", newline="") as file:
        fieldnames = ["holdout", "pearson_corr"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        rows = [{"holdout": holdout_set, "pearson_corr": float(r)}]
        writer.writerows(rows)   # write list of dicts
    print(r)


























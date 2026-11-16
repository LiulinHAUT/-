"""Siamese BiLSTM + Attention 中文语义匹配模型训练脚本"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np
import json
import os
from collections import Counter
from tqdm import tqdm

# ==================== 配置 ====================
class Config:
    data_dir = "lcqmc"
    max_vocab = 30000
    max_len = 40
    embed_dim = 256
    hidden_dim = 128
    num_layers = 2
    dropout = 0.3
    attention_dim = 64
    mlp_dim = 128
    batch_size = 128
    epochs = 20
    lr = 0.001
    weight_decay = 1e-4
    patience = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "output"
    figures_dir = "figures"

config = Config()
os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.figures_dir, exist_ok=True)

# ==================== 数据加载 ====================
def load_data(filepath):
    sentences_a, sentences_b, labels = [], [], []
    with open(filepath, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                sentences_a.append(parts[0])
                sentences_b.append(parts[1])
                labels.append(int(parts[2]))
    return sentences_a, sentences_b, labels

def tokenize(text):
    return list(jieba.cut(text.strip()))

def build_vocab(sentences_a, sentences_b, max_size):
    counter = Counter()
    for a, b in zip(sentences_a, sentences_b):
        counter.update(tokenize(a))
        counter.update(tokenize(b))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab

class LCQMCDataset(Dataset):
    def __init__(self, sentences_a, sentences_b, labels, vocab, max_len):
        self.data = []
        for a, b, l in zip(sentences_a, sentences_b, labels):
            ids_a = [vocab.get(w, 1) for w in tokenize(a)][:max_len]
            ids_b = [vocab.get(w, 1) for w in tokenize(b)][:max_len]
            len_a = min(len(ids_a), max_len)
            len_b = min(len(ids_b), max_len)
            ids_a += [0] * (max_len - len(ids_a))
            ids_b += [0] * (max_len - len(ids_b))
            self.data.append((ids_a, len_a, ids_b, len_b, l))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, la, b, lb, l = self.data[idx]
        return (
            torch.tensor(a, dtype=torch.long),
            torch.tensor(la, dtype=torch.long),
            torch.tensor(b, dtype=torch.long),
            torch.tensor(lb, dtype=torch.long),
            torch.tensor(l, dtype=torch.float32),
        )

# ==================== 模型定义 ====================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.att = nn.Linear(hidden_dim * 2, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, outputs, mask):
        scores = self.v(torch.tanh(self.att(outputs))).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1).unsqueeze(1)  # [B, 1, L]
        pooled = torch.bmm(weights, outputs).squeeze(1)  # [B, 2H]
        return pooled

class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 dropout, attention_dim, mlp_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = AttentionPooling(hidden_dim, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 8, mlp_dim),  # 4 concat * 2 bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),
        )

    def encode(self, ids, lengths):
        max_len = ids.size(1)
        embedded = self.embedding(ids)  # [B, L, E]
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=max_len
        )
        mask = (ids != 0).float()  # [B, L]
        vec = self.attention(outputs, mask)
        return vec

    def forward(self, ids_a, len_a, ids_b, len_b):
        vec_a = self.encode(ids_a, len_a)
        vec_b = self.encode(ids_b, len_b)
        feat = torch.cat([vec_a, vec_b, torch.abs(vec_a - vec_b), vec_a * vec_b], dim=-1)
        feat = self.dropout(feat)
        logits = self.classifier(feat).squeeze(-1)
        return logits

# ==================== 训练 ====================
def compute_accuracy(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == labels).float().mean().item()

def train():
    print(f"Device: {config.device}")

    # 加载数据
    print("Loading data...")
    train_a, train_b, train_y = load_data(f"{config.data_dir}/train.tsv")
    dev_a, dev_b, dev_y = load_data(f"{config.data_dir}/dev.tsv")

    # 构建词表
    print("Building vocabulary...")
    vocab = build_vocab(train_a, train_b, config.max_vocab)
    with open(f"{config.output_dir}/vocab.json", "w") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Vocab size: {len(vocab)}")

    # 数据集
    train_ds = LCQMCDataset(train_a, train_b, train_y, vocab, config.max_len)
    dev_ds = LCQMCDataset(dev_a, dev_b, dev_y, vocab, config.max_len)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # 模型
    model = SiameseLSTM(
        len(vocab), config.embed_dim, config.hidden_dim, config.num_layers,
        config.dropout, config.attention_dim, config.mlp_dim
    ).to(config.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # 记录指标
    history = {"train_loss": [], "dev_loss": [], "dev_acc": [], "lr": []}

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0

    print("\nTraining Siamese LSTM...")
    for epoch in range(1, config.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for batch in train_pbar:
            ids_a, len_a, ids_b, len_b, labels = [x.to(config.device) for x in batch]
            optimizer.zero_grad()
            logits = model(ids_a, len_a, ids_b, len_b)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Eval
        model.eval()
        dev_loss = 0.0
        dev_acc = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                ids_a, len_a, ids_b, len_b, labels = [x.to(config.device) for x in batch]
                logits = model(ids_a, len_a, ids_b, len_b)
                loss = criterion(logits, labels)
                dev_loss += loss.item()
                dev_acc += compute_accuracy(logits, labels)

        avg_dev_loss = dev_loss / len(dev_loader)
        avg_dev_acc = dev_acc / len(dev_loader)
        history["dev_loss"].append(avg_dev_loss)
        history["dev_acc"].append(avg_dev_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, dev_loss={avg_dev_loss:.4f}, "
              f"dev_acc={avg_dev_acc:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(avg_dev_acc)

        if avg_dev_acc > best_acc:
            best_acc = avg_dev_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), f"{config.output_dir}/lstm_best.pt")
        else:
            no_improve += 1

        if no_improve >= config.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest dev acc: {best_acc:.4f} at epoch {best_epoch}")
    torch.save(history, f"{config.output_dir}/lstm_history.pt")
    torch.save({"vocab": vocab, "config": config.__dict__}, f"{config.output_dir}/lstm_meta.pt")

if __name__ == "__main__":
    train()

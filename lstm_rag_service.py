"""
LSTM 语义检索服务 —— 基于训练好的 Siamese LSTM 模型
替换原 rag_service.py 中的 TF-IDF 检索，提供语义级别的知识检索

使用方式:
    from app.services.lstm_rag_service import LSTMRetriever

    retriever = LSTMRetriever()
    retriever.build_index(chunks)                     # 构建向量索引
    results = retriever.search(query, top_k=5)         # 语义检索
"""

import json
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

import jieba

# ==================== 模型文件路径 ====================
_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'output')
_MODEL_PATH = os.path.join(_MODEL_DIR, 'lstm_best.pt')
_VOCAB_PATH = os.path.join(_MODEL_DIR, 'vocab.json')
_META_PATH = os.path.join(_MODEL_DIR, 'lstm_meta.pt')

# ==================== 模型参数 ====================
_CONFIG = {
    'vocab_size': 30000,
    'embed_dim': 256,
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'attention_dim': 64,
    'mlp_dim': 128,
    'max_len': 40,
}

# ==================== 模型定义（与训练时保持一致）====================


class AttentionPooling(nn.Module):
    """注意力池化层"""
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.att = nn.Linear(hidden_dim * 2, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.v(torch.tanh(self.att(outputs))).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1).unsqueeze(1)
        return torch.bmm(weights, outputs).squeeze(1)


class SiameseLSTMEncoder(nn.Module):
    """
    Siamese LSTM 编码器（仅推理用，不含分类器）
    将文本映射为 256 维语义向量
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float, attention_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = AttentionPooling(hidden_dim, attention_dim)
        self.max_len = _CONFIG['max_len']

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        max_len = ids.size(1)
        embedded = self.embedding(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=max_len)
        mask = (ids != 0).float()
        return self.attention(outputs, mask)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """返回归一化的句向量"""
        vec = self.encode(ids, lengths)
        return F.normalize(vec, p=2, dim=1)


# ==================== 文本预处理 ====================


def _tokenize(text: str) -> List[str]:
    """jieba 中文分词"""
    return list(jieba.cut(text.strip()))


def _text_to_tensor(text: str, vocab: dict, max_len: int = 40) -> tuple:
    """将文本转为模型输入张量"""
    tokens = _tokenize(text)
    ids = [vocab.get(w, 1) for w in tokens][:max_len]  # 1 = <UNK>
    real_len = max(len(ids), 1)
    ids += [0] * (max_len - len(ids))  # 0 = <PAD>
    return torch.tensor([ids], dtype=torch.long), torch.tensor([real_len], dtype=torch.long)


# ==================== LSTM 检索器 ====================


class LSTMRetriever:
    """
    基于 Siamese LSTM 的语义检索器

    用法:
        retriever = LSTMRetriever()
        retriever.build_index(chunks)              # chunks 是文本列表
        results = retriever.search('如何提高成绩', top_k=5)
        # 返回: [{'chunk_index': 3, 'content': '...', 'score': 0.92}, ...]
    """

    def __init__(self):
        self._vocab: Optional[dict] = None
        self._model: Optional[SiameseLSTMEncoder] = None
        self._device: Optional[torch.device] = None
        self._chunk_vectors: Optional[np.ndarray] = None
        self._chunks: Optional[List[str]] = None
        self._loaded = False

    def _ensure_loaded(self):
        """延迟加载模型（首次使用时加载）"""
        if self._loaded:
            return

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载词表（先用UTF-8，不行换GBK）
        try:
            with open(_VOCAB_PATH, 'r', encoding='utf-8') as f:
                self._vocab = json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            with open(_VOCAB_PATH, 'r', encoding='gbk') as f:
                self._vocab = json.load(f)

        # 尝试从 meta 文件加载配置，否则用默认值
        cfg = dict(_CONFIG)
        try:
            if os.path.exists(_META_PATH):
                meta = torch.load(_META_PATH, map_location='cpu', weights_only=False)
                meta_cfg = meta.get('config', {})
                if isinstance(meta_cfg, dict) and 'embed_dim' in meta_cfg:
                    for k in ['embed_dim', 'hidden_dim', 'num_layers',
                               'dropout', 'attention_dim', 'mlp_dim']:
                        if k in meta_cfg:
                            cfg[k] = meta_cfg[k]
        except Exception:
            pass  # 用默认值

        # 构建模型
        model = SiameseLSTMEncoder(
            vocab_size=cfg['vocab_size'],
            embed_dim=cfg['embed_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout'],
            attention_dim=cfg['attention_dim'],
        ).to(self._device)

        # 加载权重
        state_dict = torch.load(_MODEL_PATH, map_location=self._device, weights_only=False)
        # 只加载 encoder 部分的权重（去掉 classifier 前缀）
        encoder_state = {k: v for k, v in state_dict.items()
                         if k.startswith('embedding') or k.startswith('lstm')
                         or k.startswith('attention')}
        model.load_state_dict(encoder_state, strict=False)
        model.eval()

        self._model = model
        self._loaded = True
        print(f'[LSTMRetriever] 模型加载完成 | device={self._device} '
              f'| vocab={len(self._vocab)} | params={sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    def encode(self, texts: List[str]) -> np.ndarray:
        """将文本列表编码为句向量矩阵，返回 shape=(len(texts), 256)"""
        self._ensure_loaded()
        vectors = []
        with torch.no_grad():
            for text in texts:
                ids, lens = _text_to_tensor(text, self._vocab, _CONFIG['max_len'])
                ids, lens = ids.to(self._device), lens.to(self._device)
                vec = self._model(ids, lens)
                vectors.append(vec.cpu().numpy()[0])
        return np.array(vectors)

    def build_index(self, chunks: List[str]):
        """为所有知识分块构建向量索引（一次性编码，后续检索复用）"""
        self._ensure_loaded()
        self._chunks = chunks
        self._chunk_vectors = self.encode(chunks)
        print(f'[LSTMRetriever] 索引构建完成 | {len(chunks)} 个分块 | 向量维度={self._chunk_vectors.shape[1]}')

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[dict]:
        """
        语义检索：返回与查询最相似的 top_k 个分块

        Args:
            query: 查询文本
            top_k: 返回结果数，默认 5
            min_score: 最低相似度阈值，默认 0

        Returns:
            [{'chunk_index': int, 'content': str, 'score': float}, ...]
        """
        self._ensure_loaded()

        if not self._chunks or self._chunk_vectors is None:
            return []

        # 编码查询向量
        query_vec = self.encode([query])

        # 余弦相似度排序
        scores = cosine_similarity(query_vec, self._chunk_vectors)[0]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:top_k]:
            if score < min_score:
                break
            results.append({
                'chunk_index': idx,
                'content': self._chunks[idx],
                'score': round(float(score), 4),
            })

        return results

    def search_top_k(self, query_vec: np.ndarray, chunk_vectors: np.ndarray,
                     chunks: List[str], top_k: int = 5, min_score: float = 0.0) -> List[dict]:
        """
        兼容接口：与 rag_service.search_top_k() 保持相同签名
        用于在现有代码中直接替换调用
        """
        self._ensure_loaded()

        if len(chunks) == 0:
            return []

        scores = cosine_similarity(query_vec, chunk_vectors)[0]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:top_k]:
            if score < min_score:
                break
            results.append({
                'chunk_index': idx,
                'content': chunks[idx],
                'score': round(float(score), 4),
            })

        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def embedding_dim(self) -> int:
        return 256


# ==================== 全局单例 ====================
_retriever: Optional[LSTMRetriever] = None


def get_lstm_retriever() -> LSTMRetriever:
    """获取全局 LSTMRetriever 单例"""
    global _retriever
    if _retriever is None:
        _retriever = LSTMRetriever()
    return _retriever

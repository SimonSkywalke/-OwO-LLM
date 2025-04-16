import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    # d_model：嵌入向量的维度（模型维度）
    # max_seq_len：最大序列长度，默认为80
    # dropout：随机丢弃率，用于防止过拟合
    def __init__(self, d_model, max_seq_len=80, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # 根据pos和i创建一个形状为(max_seq_len, d_model)常量PE矩阵 pos：位置索引 i维度索引
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        # 增加一个批处理维度，将形状从 [max_seq_len, d_model] 变为 [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len, :], requires_grad=False)
        return self.dropout(x)
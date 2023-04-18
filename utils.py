import torch
import torch.nn as nn
import numpy as np
import copy


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.w = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        return self.w * (x - mean) / np.sqrt(var + self.eps) + self.b


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.layerNorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.layerNorm(x + sublayer(x)))



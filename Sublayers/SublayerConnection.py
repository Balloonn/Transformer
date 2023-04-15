import torch
import torch.nn as nn
import numpy as np

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

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout = 0.1):
        super(SublayerConnection, self).__init__()
        self.layerNorm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.layerNorm(x + sublayer(x)))


import torch
import torch.nn as nn
from LayerNorm import LayerNorm

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout = 0.1):
        super(SublayerConnection, self).__init__()
        self.layerNorm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.layerNorm(x + sublayer(x)))


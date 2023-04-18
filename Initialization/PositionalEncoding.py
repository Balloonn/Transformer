import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        if d_model % 2 != 0:
            raise ValueError(f"维度为奇数，无法生成位置编码")
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos.float() * div_term)
        pe[:, 1::2] = torch.cos(pos.float() * div_term)
        pe = pe.unsqueeze(1)

        self.register_buffer('pe', pe)
        self.dim = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb):
        # emb:  seq_len * batch * dim
        emb = emb * np.sqrt(self.dim)
        emb = emb + self.pe[:emb.size(0)]
        emb = self.dropout(emb)
        return emb


if __name__ == '__main__':
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    plt.plot(np.arange(100), y[:, 0, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()

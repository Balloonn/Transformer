import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def self_attention(q, k, v, dropout=None, mask=None):
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        mask.cuda()
        scores = scores.masked_filled(mask == 0, -1e9)
    self_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        self_attn = dropout(self_attn, p=dropout)
    return torch.matmul(self_attn, v), self_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.attn_softmax = None

    def forward(self, q, k, v, mask=None):
        # q: batch * seq_len * dim
        if mask is not None:
            mask.unsqueeze(1)
        batch = q.size(0)
        q = self.linear_q(q).view(batch, -1, self.head, self.d_k).transpose(1, 2)
        k = self.linear_k(k).view(batch, -1, self.head, self.d_k).transpose(1, 2)
        v = self.linear_k(v).view(batch, -1, self.head, self.d_k).transpose(1, 2)
        x, self.attn_softmax = self_attention(q, k, v, dropout=self.dropout, mask=mask)
        x = x.transpose(1,2).contiguous().view(batch, -1, self.head * self.d_k)
        return self.linear_out(x)



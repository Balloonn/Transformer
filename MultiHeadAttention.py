import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import clones


def self_attention(q, k, v, dropout=None, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    self_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        self_attn = dropout(self_attn)
    return torch.matmul(self_attn, v), self_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn_softmax = None

    def forward(self, q, k, v, mask=None):
        # q: batches * batch_size * seq_len
        if mask is not None:  # batches * batch_size * seq_len
            mask = mask.unsqueeze(1)  # batches * head_num * batch_size * d_k
        batches = q.size(0)
        q = self.linear_q(q).view(batches, -1, self.head, self.d_k).transpose(1, 2)
        k = self.linear_k(k).view(batches, -1, self.head, self.d_k).transpose(1, 2)
        v = self.linear_k(v).view(batches, -1, self.head, self.d_k).transpose(1, 2)
        x, self.attn_softmax = self_attention(q, k, v, dropout=self.dropout, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batches, -1, self.head * self.d_k)
        return self.linear_out(x)



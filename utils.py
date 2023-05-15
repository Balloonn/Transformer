import time
import torch
import torch.nn as nn
import numpy as np
import copy
from torch.autograd import Variable


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


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


def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0  # tokens in 50 batches
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_pred, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d, Loss: %f, Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_trg_in_batch


# def batch_size_fn(new, count, sofar):
#     global max_src_in_batch, max_trg_in_batch
#     if count == 1:
#         max_src_in_batch = 0
#         max_trg_in_batch = 0
#     max_src_in_batch = max(max_src_in_batch, len(new.src))
#     max_trg_in_batch = max(max_trg_in_batch, len(new.trg))
#     src_elements = count * max_src_in_batch
#     trg_elements = count * max_trg_in_batch
#     return max(src_elements, trg_elements)


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size  # seq_len
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size()  # batch_size * seq_len
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))

        true_dist.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))




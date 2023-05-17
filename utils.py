import time
import torch
import torch.nn as nn
import numpy as np
import copy
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F


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
        std = x.std(-1, keepdim=True)
        return self.w * (x - mean) / (std + self.eps) + self.b


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout(F.relu(self.w_1(x)))
        output = self.w_2(inter)
        return output


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.layerNorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layerNorm(x)))
        # return self.dropout(self.layerNorm(x + sublayer(x)))


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


# global max_src_in_batch, max_trg_in_batch
#
#
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

class NoamOpt:
    def __init__(self, d_model, factor, warmup_step, optimizer):
        self.d_model = d_model
        self.factor = factor
        self.warmup_step = warmup_step
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_step ** (-1.5)))


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size  # vocab_num(original vocab_num + padding signal)
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()  # seq_len * vocab_num
        true_dist.fill_(self.smoothing / (self.size - 2))  # one for target, one for padding signal

        true_dist.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


if __name__ == "__main__":
    # 针对不同模型大小和优化超参数的曲线示例。
    # 设置三个不同的模型尺寸，最大学习率上升步阈值
    opts = [NoamOpt(512, 1, 4000, None),
            NoamOpt(512, 1, 8000, None),
            NoamOpt(256, 1, 4000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.show()

    # 测试 label smoothing.
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    v = crit(Variable(predict.log()), Variable(torch.LongTensor([2, 1, 0])))
    print(predict.log())
    print(crit.true_dist)
    # 显示系统预期的目标分布.
    plt.imshow(crit.true_dist)
    plt.show()

    # 标签平滑实际上开始惩罚模型，如果它对给定的选择非常自信。
    crit = LabelSmoothing(5, 0, 0.1)

    def loss(x):
        d = x + 3 * 1
        predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d], ])
        predict += 1e-8
        print("predict:", predict)
        print(crit(Variable(predict.log()), Variable(torch.LongTensor([1]))).item())
        return crit(Variable(predict.log()), Variable(torch.LongTensor([1]))).item()
    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.show()


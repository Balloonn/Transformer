import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, d_model, trg_vocab):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, trg_vocab)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)

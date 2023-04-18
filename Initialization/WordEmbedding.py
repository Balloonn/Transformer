import torch.nn as nn
import torch


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(WordEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(self.d_model)

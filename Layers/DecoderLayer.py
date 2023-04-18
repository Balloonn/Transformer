import torch.nn as nn
from clone import clones
from Sublayers import SublayerConnection


class DecoderLayer(nn.Module):
    def __init__(self, d_model, attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connections = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask=None, trg_mask=None):
        first_x = self.sublayer_connections[0](x, lambda first_x_attn: self.attn(x, x, x, src_mask))
        second_x = self.sublayer_connections[1](first_x, lambda second_x_attn: self.attn(x, memory, memory, trg_mask))
        return self.sublayer_connections[2](second_x, self.feed_forward())

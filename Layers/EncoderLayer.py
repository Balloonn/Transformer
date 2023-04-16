import torch.nn as nn
from clone import clones
from Sublayers.SublayerConnection import SublayerConnection


class EncoderLayer(nn.Module):
    def __init__(self, d_model, attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connections = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        first_x = self.sublayer_connections[0](x, lambda x_attn: self.attn(x, x, x, mask))
        return self.sublayer_connections[1](first_x, self.feed_forward())
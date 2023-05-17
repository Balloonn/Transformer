import torch.nn as nn
from utils import clones, SublayerConnection, subsequent_mask, LayerNorm
import matplotlib.pyplot as plt


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer_connections = clones(SublayerConnection(d_model, dropout), 3)
        self.d_model = d_model

    def forward(self, x, memory, src_mask=None, trg_mask=None):
        first_x = self.sublayer_connections[0](x, lambda first_x_attn: self.self_attn(first_x_attn, first_x_attn, first_x_attn, trg_mask))
        second_x = self.sublayer_connections[1](first_x, lambda second_x_attn: self.src_attn(second_x_attn, memory, memory, src_mask))
        return self.sublayer_connections[2](second_x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, n, decoder_layer):
        super(Decoder, self).__init__()
        self.decoder_layers = clones(decoder_layer, n)
        self.layer_norm = LayerNorm(decoder_layer.d_model)

    def forward(self, x, memory, src_mask, trg_mask):
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, trg_mask)
        return self.layer_norm(x)


if __name__ == '__main__':
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20)[0])
    plt.show()

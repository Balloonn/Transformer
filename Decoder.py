import torch.nn as nn
from Layers.clone import clones


class Decoder(nn.Module):
    def __init__(self, n, decoder_layer):
        super(Decoder, self).__init__()
        self.decoder_layers = clones(decoder_layer, n)

    def forward(self, x, memory, trg_mask):
        for layer in self.decoder_layers:
            x = layer(x, memory, trg_mask)
        return x

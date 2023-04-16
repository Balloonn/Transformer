import torch.nn as nn
from Layers.clone import clones


class Encoder(nn.Module):
    def __init__(self, n, encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layers = clones(encoder_layer, n)

    def forward(self, x, src_mask):
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

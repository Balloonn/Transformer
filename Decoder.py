import torch
import torch.nn as nn
from Layers.clone import clones
import numpy as np
import matplotlib.pyplot as plt


class Decoder(nn.Module):
    def __init__(self, n, decoder_layer):
        super(Decoder, self).__init__()
        self.decoder_layers = clones(decoder_layer, n)

    def forward(self, x, memory, trg_mask):
        for layer in self.decoder_layers:
            x = layer(x, memory, trg_mask)
        return x


def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


if __name__ == '__main__':
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20)[0])
    plt.show()

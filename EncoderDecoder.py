import torch.nn as nn
from Sublayers import MultiHeadAttention, FeedForward
from Initialization import PositionalEncoding, WordEmbedding
from Layers import EncoderLayer, DecoderLayer
import Encoder, Decoder, Generator
import copy as c


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, trg_embed, pos_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.pos_embed = pos_embed
        self.generator = generator

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.pos_embed(x)
        x = self.encoder(x, src_mask)
        return x

    def decode(self, trg, memory, trg_mask):
        x = self.trg_embed(trg)
        x = self.pos_embed(x)
        x = self.decoder(x, memory, trg_mask)
        return x

    def forward(self, src, trg, mask):
        src_mask, trg_mask = mask
        encoder_outputs = self.encode(src, src_mask)
        decoder_outputs = self.decode(trg, encoder_outputs, trg_mask)
        outputs = self.generator(decoder_outputs)
        return outputs

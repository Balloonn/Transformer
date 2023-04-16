import torch.nn as nn
from Sublayers import MultiHeadAttention, FeedForward
from Initialization import PositionalEncoding, WordEmbedding
from Layers import EncoderLayer, DecoderLayer
import Encoder, Decoder, Generator
import copy as c


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, d_ff, n_heads, n_layers, dropout, device='cuda'):
        super(Transformer, self).__init__()
        self.device = device
        attn = MultiHeadAttention(n_heads, d_model, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        self.src_embed = WordEmbedding(src_vocab, d_model)
        self.trg_embed = WordEmbedding(trg_vocab, d_model)
        self.pos_embed = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(n_layers, EncoderLayer(d_model, c.deepcopy(attn), c.deepcopy(feed_forward), dropout))
        self.decoder = Decoder(n_layers, DecoderLayer(d_model, c.deepcopy(attn), c.deepcopy(feed_forward), dropout))
        self.generator = Generator(d_model, trg_vocab)

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

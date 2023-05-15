from MultiHeadAttention import MultiHeadAttention
from utils import FeedForward
from PositionalEncoding import PositionalEncoding
from WordEmbedding import WordEmbedding
from Encoder import Encoder, EncoderLayer
from Decoder import Decoder, DecoderLayer
from Generator import Generator
import copy as c
import torch.nn as nn


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

    def forward(self, src, trg, src_mask, trg_mask):
        encoder_outputs = self.encode(src, src_mask)
        decoder_outputs = self.decode(trg, encoder_outputs, trg_mask)
        outputs = self.generator(decoder_outputs)
        return outputs


def make_model(src_vocab, trg_vocab, d_model=512, d_ff=2048, n_heads=8, n_layers=6, dropout=0.1):
    attn = MultiHeadAttention(n_heads, d_model, dropout)
    feed_forward = FeedForward(d_model, d_ff, dropout)
    model = EncoderDecoder(
        Encoder(n_layers, EncoderLayer(d_model, c.deepcopy(attn), c.deepcopy(feed_forward), dropout)),
        Decoder(n_layers, DecoderLayer(d_model, c.deepcopy(attn), c.deepcopy(feed_forward), dropout)),
        WordEmbedding(src_vocab, d_model),
        WordEmbedding(trg_vocab, d_model),
        PositionalEncoding(d_model, dropout),
        Generator(d_model, trg_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

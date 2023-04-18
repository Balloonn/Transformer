from Sublayers import MultiHeadAttention, FeedForward
from Initialization import PositionalEncoding, WordEmbedding
from Layers.EncoderLayer import EncoderLayer
from Layers.DecoderLayer import DecoderLayer
import Encoder, Decoder, Generator, EncoderDecoder
import copy as c


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

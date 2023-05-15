from utils import subsequent_mask
from torch.autograd import Variable


class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src  # batch_size * seq_len
        self.src_mask = (src != pad).unsqueeze(-2)  # batches * batch_size * seq_len
        if trg is not None:
            self.trg_deco = trg[:, :-1]
            self.trg_pred = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg_deco, pad)
            self.ntokens = (self.trg_pred != pad).data.sum()

    @staticmethod
    def make_std_mask(trg, pad):
        trg_mask = (trg != pad).unsqueeze(-2)  # batches * batch_size * seq_len
        trg_mask = trg_mask & Variable(subsequent_mask(trg_mask.size(-1)).type_as(trg_mask.data))
        return trg_mask


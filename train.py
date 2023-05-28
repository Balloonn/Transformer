from utils import *
import torch
import numpy as np
from torch.autograd import Variable
from Transformer import make_model
import os


class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src  # batch_size * seq_len
        self.src_mask = torch.Tensor((src != pad)).int().unsqueeze(-2)  # batches * batch_size * seq_len
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_pred = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_pred != pad).data.sum()

    @staticmethod
    def make_std_mask(trg, pad):
        trg_mask = torch.Tensor((trg != pad)).int().unsqueeze(-2)  # batches * batch_size * seq_len
        trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).type_as(trg_mask.data))
        return trg_mask


def data_gen(V, batches, batch):
    for i in range(batches):
        data = torch.from_numpy(np.random.randint(0, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        trg = Variable((data[:, 1:]+3) % 11, requires_grad=False)
        yield Batch(src, trg, 0)


class LossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(Variable(ys), memory, src_mask,
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


if __name__ == "__main__":
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, n_layers=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    losslast = float('inf')
    for epoch in range(100):
        model.train()
        run_epoch(data_gen(V, 50, 100), model, LossCompute(model.generator, criterion, model_opt))
        model.eval()
        loss = run_epoch(data_gen(V, 10, 100), model, LossCompute(model.generator, criterion, model_opt))
        if loss < losslast:
            if not os.path.exists("parameters"):
                os.mkdir("parameters")
            torch.save(model.state_dict(), 'parameters/params.pkl')
            print("update best model")
            losslast = loss
        print("epoch: %d, loss: %f" % (epoch, loss))

    model.eval()
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    model.load_state_dict(torch.load('parameters/params.pkl'))
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=4))

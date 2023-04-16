import copy
import torch.nn as nn


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

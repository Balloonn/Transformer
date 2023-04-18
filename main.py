import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from Initialization import PositionalEncoding


plt.figure(figsize=(15,5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
plt.show()

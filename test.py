from matplotlib.pyplot import axes
import torch.nn as nn
import torch

a = torch.rand([2, 20, 1, 1])
print(a.sum(1))
s = nn.Softmax(dim=1)
y = s(a)
print(y.sum(1))


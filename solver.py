"""
.. module:: solver
   :synopsis: This file contains ...
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import layer_norm

def sgd(w0, X, Y, var, batch_size, epochs):
    diffs_norm = layer_norm(X - Y, square=True)

    if var is None:
        var = torch.quantile(diffs_norm, 0.1, interpolation='midpoint').item()

    gauss = torch.exp(-diffs_norm / (2. * var ))
    dataloader = Dataloader(zip(X, Y, gauss), batch_size=batch_size, shuffle=False)
    f_history = torch.zeros(epochs)

    LGK = LinearGaussianKernel(w0, var).to(X.device)
    optimiser = optim.Adam(LGK.parameters(), lr=1e-2)

    for i in range(epochs):
        running_f = 0.

        for x, y, k in dataloader:
            k_approx = LGK(x, y)
            f = torch.square(k - k_approx).mean()
            running_f += f.item()

            optimiser.zero_grad()
            f.backward()
            optimiser.step()

        f_history[i] = running_f

    print(f_history.view(-1, 10))

    return *LGK.parameters(), var


class SemiLGK(nn.Module):
    def __init__(self, w0, var):
        super().__init__()
        self.w0 = nn.Parameter(w0)
        self.var = var

    def forward(self, X):
        diffs = X.unsqueeze(1) - self.w0
        diffs_norm = layer_norm(diffs, squared=True)
        return torch.exp(-diffs_norm / self.var)


class LinearGaussianKernel(nn.Module):
    def __init__(self, w0, var):
        super().__init__()
        self.slgk = SemiLGK(w0, var)
        self.eta = nn.Parameter(torch.zeros(len(w0)))
        self.var = var

    def forward(self, X, Y):
        return torch.matmul(self.eta * self.slgk(X), self.slgk(Y).t())

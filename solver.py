"""
.. module:: solver
   :synopsis: This file contains ...
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import layer_norm

def _dataloader(X, Y, K, batch_size):
    n = len(X)
    n_batches = int(math.ceil(n / batch_size))
    for i in range(n_batches):
        b = i * batch_size
        e = min(b + batch_size, n)
        yield X[b:e], Y[b:e], K[b:e]


def sgd(w0, X, Y, lr, var, batch_size, epochs):
    diffs_norm = layer_norm(X - Y, squared=True)

    if var is None:
        var = torch.quantile(
            diffs_norm,
            q=0.1,
            dim=0,
            keepdim=False,
            interpolation='midpoint'
        ).item()

    gauss = torch.exp(-diffs_norm / (2. * var ))
    n = len(X)

    f_history = torch.zeros(epochs)

    LGK = LinearGaussianKernel(w0, var).to(X.device)
    optimiser = optim.Adam(LGK.parameters(), lr=lr)

    verbose_interval = epochs // 10
    print()

    for i in range(epochs):
        running_f = 0.

        for x, y, k in _dataloader(X, Y, gauss, batch_size):
            k_approx = LGK(x, y)
            f = torch.square(k - k_approx).mean()
            running_f += f.item()

            optimiser.zero_grad()
            f.backward()
            optimiser.step()

        f_history[i] = running_f

        if i % verbose_interval == verbose_interval-1:
            print(f'\rSGD epoch {i+1}/{epochs} | Gap: {running_f:.4f}',
                  end='', flush=True)

    print('\033[1K\033[F\033[22C', end='', flush=True)

    # print(f_history.view(-1, 10))

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
        self.log_eta = nn.Parameter(torch.zeros(len(w0)))

    def forward(self, X, Y):
        return torch.bmm(
            torch.exp(self.log_eta) * self.slgk(X).unsqueeze(1),
            self.slgk(Y).unsqueeze(2)
        ).squeeze()

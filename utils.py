"""
.. module:: utils
   :synopsis: This file contains ...
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

# import numpy as np
import torch

def batch_norm(X, squared=False):
    norms = torch.einsum('bhwc,bhwc->b', X, X)

    if not squared:
        torch.sqrt(norms, norms)
    return norms


def euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None,
                        squared=False):
    if X_norm_squared is None:
        X_norm_squared = batch_norm(X, squared=True)
    XX = X_norm_squared.unsqueeze(1)

    if Y_norm_squared is None:
        Y_norm_squared = batch_norm(Y, squared=True)
    YY = Y_norm_squared.unsqueeze(0)

    distances = -2. * torch.einsum('xhwc,yhwc->xy', X, Y)
    distances += XX
    distances += YY
    torch.maximum(distances, 0, out=distances)

    if not squared:
        torch.sqrt(distances, distances)

    return distances

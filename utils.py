"""
.. module:: utils
   :synopsis: This file contains ...
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

import torch
import torch.nn.functional as F

def layer_norm(X, squared=False):
    norms = torch.einsum('...hwc,...hwc->...', X, X)

    if not squared:
        norms = torch.sqrt(norms)
    return norms


def layer_normalise(X, eps=1e-5):
    norms = layer_norm(X)
    return X / torch.clip(norms, min=eps).view(-1, 1, 1, 1), norms


def euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None,
                        squared=False):
    if X_norm_squared is None:
        X_norm_squared = layer_norm(X, squared=True)
    XX = X_norm_squared.unsqueeze(1)

    if Y_norm_squared is None:
        Y_norm_squared = layer_norm(Y, squared=True)
    YY = Y_norm_squared.unsqueeze(0)

    distances = -2. * torch.einsum('xhwc,yhwc->xy', X, Y)
    distances += XX
    distances += YY
    torch.clip(distances, min=0., out=distances)

    if not squared:
        torch.sqrt(distances, out=distances)

    return distances


def dynamic_partition(data, partitions, num_partitions=None):
    assert len(partitions.shape) == 1, 'Only 1D partitions supported'
    assert data.shape[0] == partitions.shape[0], \
        'Partitions requires the same size as data'

    if num_partitions is None:
        num_partitions = max(torch.unique(partitions))

    return [data[partitions == i] for i in range(num_partitions)]


def extract_sq_patches(x, size, stride, same=True):
    """x : (batch, h, w, c) """
    if same:
        pad = size - 1
        x = F.pad(x, (0, 0, pad, pad, pad, pad))

    patches = x.unfold(1, size, stride).unfold(2, size, stride)
    return patches.contiguous().view(-1, size, size, x.shape[-1])


def gaussian_window(n, var, dtype, device):
    if n <= 1:
        return torch.ones(1, 1, dtype=dtype, device=device)
    w = torch.arange(n, dtype=dtype, device=device) - (n - 1.) / 2.
    ww = torch.stack(torch.meshgrid(w, w, indexing='ij'), dim=-1)
    ww = torch.einsum('ijk,ijk->ij', ww, ww)  # square of the norm
    return torch.exp(-ww / var)

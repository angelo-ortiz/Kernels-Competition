"""
.. module:: utils
   :synopsis: This file contains ...
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

import torch

def layer_norm(X, squared=False):
    norms = torch.einsum('...hwc,...hwc->...', X, X)

    if not squared:
        norms = torch.sqrt(norms)
    return norms


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


def extract_sq_patches(x, size, stride):  # TODO: need for padding in the patches???
    """x : (batch, h, w, c) """
    patches = x.unfold(1, size, stride).unfold(2, size, stride)
    return patches.contiguous().view(-1, size, size, x.shape[-1])


def gaussian_window(n, var, device):
    if n <= 1:
        return torch.ones(1, 1, device=device)
    w = torch.arange(n, device=device) - (n - 1.) / 2.
    ww = torch.stack(torch.meshgrid(w, w, indexing='ij'), dim=-1)
    ww = torch.einsum('ijk,ijk->ij', ww, ww)  # square of the norm
    return torch.exp(-ww / var)

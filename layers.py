"""
.. module:: layers
   :synopsis: This file contains ...
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import feats_norm
from kmeans import k_means
# from solver import l_bfgs_b, sgd

MIN_NORM = 1e-3

def extract_sq_patches(x, size, stride):
    """x : (batch, h, w, c) """
    patches = x.unfold(1, size, stride).unfold(2, size, stride)
    return patches.contiguous().view(-1, size, size, x.shape[-1])

def gaussian_window(n, var, device):
    if n <= 1:
        return torch.ones(1, 1, device=device)
    w = torch.arange(n, device=device) - (n - 1.) / 2.
    ww = torch.stack([w, w.t()], dim=-1)
    ww = torch.einsum('ijk,ijk->ij', ww, ww)  # square of the norm
    return torch.exp(-w / var)


class CKN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def backward(self):
        pass

class CKNLayer:
    def __init__(self, out_filters, patch_size, subsample_factor,
                 solver_samples, var=None):
        self.out_filters = out_filters  # p_k
        self.patch_size = patch_size  # sqrt(|P'_k|)
        self.subsample_factor = subample_factor
        self.solver_samples = solver_samples
        self.var = var

    def train(self, input_maps):
        """TODO

        Parameters
        ----------
        input_maps : Tensor[batch, h, w, c]
            The input maps `xi_{k-1}`.
        """
        if input_maps.shape[-1] > 2:  # high dimension
            patches = extract_sq_patches(input_maps, self.patch_size, 1)
            # patches = patches.view(
            #     -1, np.prod(patches.shape[1:])
            # )

            indices = torch.randperm(patches.shape[0])
            sampled_indices = torch.randint(len(indices),
                                            (2, self.solver_samples))

            X, Y = patches[indices[sampled_indices]]
            w0, _, _ = k_means(torch.cat((X, Y), dim=0), self.out_filters)
            self.w, self.eta, self.var = sgd(w0, X, Y, self.var)
        # if self.sigma is None:
        #     self.sigma = q_01(...)

    def forward(self, input_maps):
        n = input_maps.shape[0]
        if input_maps.shape[-1] <= 2:  # low dimension
            pass
        else:  # high dimension
            patches = extract_sq_patches(input_maps, self.patch_size, 1)
            norms = feats_norm(patches)
            patches_norm = patches \
                / torch.maximum(MIN_NORM, norms).view(-1, 1, 1, 1)

            diffs = patches_norm.unsqueeze(1) - self.w
            diffs_norm = torch.einsum('pohwi,ohwi->po', diffs, diffs)

            act_maps = torch.exp(-diffs_norm/var) * torch.sqrt(self.eta)
            act_maps *= norms.unsqueeze(1) # batch * h * w, p_k

            gauss_window = gaussian_window( # h', w'
                n=self.subsample_factor,
                var=self.subsample_factor**2,
                device=input_maps.device
            )

            patch_dim = np.sqrt(act_maps.shape[0]/n)
            assert patch_dim == int(patch_dim)
            patch_dim = int(patch_dim)

            act_maps = act_maps.view(n, patch_dim, patch_dim, self.out_filters)
            act_maps = act_maps.permute(0, 3, 1, 2).contiguous()
            act_maps = act_maps.view(-1, 1, patch_dim, patch_dim)

            output_maps = F.conv2d(
                act_maps,  # batch * p_k, 1, h, w
                gauss_window.view(1, 1, *gaussian_window.shape),  # 1, 1, h', w'
                stride=self.subsample_factor
            )
            output_maps = out_maps.view(
                n,
                self.out_filters,
                *output_maps.shape[2:]
            ).permute(0, 2, 3, 1).contiguous()

            return output_maps * np.sqrt(2./np.pi) # batch, p_k, h", w"

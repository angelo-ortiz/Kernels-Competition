"""
.. module:: layers
   :synopsis: This file contains ...
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import layer_norm, extract_sq_patches, gaussian_window
from kmeans import k_means
from solver import sgd# , l_bfgs_b

MIN_NORM = 1e-3

class CKN:
    def __init__(self, out_filters, patch_sizes, subsample_factors,
                 solver_samples, var=None, batch_size=50, epochs=4000):
        super().__init__()
        self.layers = []
        for i in range(len(out_filters)):
            self.layers.append(CKNLayer(
                out_filters[i],
                patch_sizes[i],
                subsample_factors[i],
                solver_samples,
                var,
                batch_size,
                epochs
            ))

    def train(self, input_maps):
        print('Training\n--------')
        output_maps = input_maps
        for i, lay in enumerate(self.layers):
            print(f'Layer {i+1}:')
            lay.train(output_maps)
            print(f'Finished layer {i+1}!')
            output_maps = lay.forward(output_maps)

    def forward(self, input_maps):
        output_maps = input_maps
        for lay in self.layers:
            output_maps = lay.forward(output_maps)
        return output_maps


class CKNLayer:
    def __init__(self, out_filters, patch_size, subsample_factor,
                 solver_samples=300000, var=None, batch_size=50, epochs=4000):
        super().__init__()
        self.out_filters = out_filters  # p_k
        self.patch_size = patch_size  # sqrt(|P'_k|)
        self.subsample_factor = subsample_factor
        self.solver_samples = solver_samples
        self.var = var
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, input_maps):
        """TODO

        Parameters
        ----------
        input_maps : Tensor[batch, h, w, c]
            The input maps `xi_{k-1}`.
        """
        if input_maps.shape[-1] > 2:  # high dimension
            patches = extract_sq_patches(input_maps, self.patch_size, 1)

            indices = torch.randperm(patches.shape[0])
            solver_samples = min(len(patches), self.solver_samples)
            sampled_indices = torch.randint(len(indices), (2, solver_samples))
            X, Y = patches[indices[sampled_indices]]

            print('Starting k-means initialisation...', end='', flush=True)
            w0, _, _ = k_means(torch.cat((X, Y), dim=0), self.out_filters)
            print('done!')
            print('Starting SGD solver...', end='', flush=True)
            eta, w, self.var = sgd(
                w0,
                X,
                Y,
                self.var,
                self.batch_size,
                self.epochs
            )
            print('done!')
            self.eta = eta.detach()
            self.w = w.detach()

    def forward(self, input_maps):
        device = input_maps.device

        assert input_maps.dim() == 4, '`input_maps` must have shape (batch, h, w, c)'
        n, _, _, c = input_maps.shape
        assert c >= 2, 'The number of channels must be at least 2'

        if c == 2:  # low dimension
            angles = torch.linspace(
                0., 2 * np.pi, self.out_filters + 1, device=device
            )[:-1]
            w = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            self.w = w.view(self.out_filters, 1, 1, 2)
            self.eta = torch.ones(self.out_filters, device=device)
            self.var = np.square(2 * np.pi / self.out_filters)
            patches = input_maps.view(-1, 1, 1, 2)
        else:  # high dimension
            patches = extract_sq_patches(input_maps, self.patch_size, 1)

        norms = layer_norm(patches)
        patches_norm = patches \
            / torch.clip(norms, min=MIN_NORM).view(-1, 1, 1, 1)

        diffs = patches_norm.unsqueeze(1) - self.w
        diffs_norm = layer_norm(diffs, squared=True)

        act_maps = torch.exp(-diffs_norm / self.var) * torch.sqrt(self.eta)
        act_maps *= norms.unsqueeze(1) # batch * h * w, p_k

        patch_dim = np.sqrt(act_maps.shape[0] / n)
        assert patch_dim == int(patch_dim), 'Only square patches supported'
        patch_dim = int(patch_dim)

        act_maps = act_maps.view(n, patch_dim, patch_dim, self.out_filters)
        act_maps = act_maps.permute(0, 3, 1, 2).contiguous()
        act_maps = act_maps.view(-1, 1, patch_dim, patch_dim)

        gauss_window = gaussian_window( # h', w'
            n=self.subsample_factor,
            var=self.subsample_factor**2,
            device=device
        )

        output_maps = F.conv2d(
            act_maps,  # batch * p_k, 1, h, w
            gauss_window.view(1, 1, *gauss_window.shape),  # 1, 1, h', w'
            stride=self.subsample_factor
        )
        output_maps = output_maps.view(
            n,
            self.out_filters,
            *output_maps.shape[2:]
        ).permute(0, 2, 3, 1).contiguous()

        return output_maps # batch, p_k, h", w"

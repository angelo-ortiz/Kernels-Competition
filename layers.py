r"""
.. module:: layers
   :synopsis: This file contains ...
.. moduleauthor:: Angelo Ortiz <github.com/angelo-ortiz>
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import layer_norm, layer_normalise, extract_sq_patches, gaussian_window
from kmeans import k_means
from solver import sgd# , l_bfgs_b

class CKN:
    def __init__(self, out_filters, patch_sizes, subsample_factors, lrs,
                 solver_samples, var=None, batch_size=50, epochs=4000):
        super().__init__()
        self.layers = []
        for i in range(len(out_filters)):
            self.layers.append(CKNLayer(
                out_filters[i],
                patch_sizes[i],
                subsample_factors[i],
                lrs[i],
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

            with torch.no_grad():
                output_maps = lay.forward(output_maps)

    def forward(self, input_maps):
        output_maps = input_maps
        with torch.no_grad():
            for lay in self.layers:
                output_maps = lay.forward(output_maps)
        return output_maps


class CKNLayer:
    def __init__(self, out_filters, patch_size, subsample_factor, lr,
                 solver_samples=300000, var=None, batch_size=50, epochs=4000):
        super().__init__()
        self.out_filters = out_filters  # p_k
        self.patch_size = patch_size  # sqrt(|P'_k|)
        self.subsample_factor = subsample_factor
        self.lr = lr
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
            with torch.no_grad():
                patches = extract_sq_patches(input_maps, self.patch_size, 1)

                indices = torch.randperm(patches.shape[0])
                solver_samples = min(len(patches), self.solver_samples)
                sampled_indices = torch.randint(len(indices), (2, solver_samples))
                X, Y = patches[indices[sampled_indices]]
                X, _ = layer_normalise(X)
                Y, _ = layer_normalise(Y)

                print('Starting k-means initialisation...', end='', flush=True)
                w0, _, _ = k_means(torch.cat((X, Y), dim=0), self.out_filters)
                print('done!')

            print('Starting SGD solver...', end='', flush=True)
            log_eta, w, self.var = sgd(
                w0,
                X,
                Y,
                self.lr,
                self.var,
                self.batch_size,
                self.epochs
            )
            print('done!')
            self.eta = torch.exp(log_eta.detach())
            self.w = w.detach()

    def forward(self, input_maps):
        device = input_maps.device

        assert input_maps.dim() == 4, '`input_maps` must have shape (batch, h, w, c)'
        n, _, _, c = input_maps.shape
        assert c >= 2, 'The number of channels must be at least 2'

        if c == 2:  # low dimension
            angles = torch.linspace(
                0., 2 * math.pi, self.out_filters + 1, device=device
            )[1:] # [:-1]
            w = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            self.w = w.view(self.out_filters, 1, 1, 2)
            self.eta = torch.ones(self.out_filters, device=device)
            angle_std = 2 * math.pi / self.out_filters
            self.var = (1 - math.cos(angle_std))**2 + math.sin(angle_std)**2
            patches = input_maps.view(-1, 1, 1, 2)
        else:  # high dimension
            patches = extract_sq_patches(input_maps, self.patch_size, 1)

        patches_norm, norms = layer_normalise(patches)

        BATCH_SIZE = 1000
        num_batches = math.ceil(patches_norm.shape[0] / BATCH_SIZE)
        diffs_norm = torch.zeros(patches_norm.shape[0], self.w.shape[0],
                                 device=patches_norm.device)
        for i in range(num_batches):
            i_min = i*BATCH_SIZE
            i_max = min(patches_norm.shape[0], (i+1)*BATCH_SIZE)
            diffs_norm[i_min:i_max] = layer_norm(
                patches_norm[i_min:i_max].unsqueeze(1) - self.w
            )
        # diffs = patches_norm.unsqueeze(1) - self.w
        # diffs_norm = layer_norm(diffs, squared=True)

        act_maps = torch.exp(-diffs_norm / self.var) * torch.sqrt(self.eta)
        act_maps *= norms.unsqueeze(1) # batch * h * w, p_k

        patch_dim = math.sqrt(act_maps.shape[0] / n)
        assert patch_dim == int(patch_dim), 'Only square patches supported'
        patch_dim = int(patch_dim)

        act_maps = act_maps.view(n, patch_dim, patch_dim, self.out_filters)
        act_maps = act_maps.permute(0, 3, 1, 2).contiguous()
        act_maps = act_maps.view(-1, 1, patch_dim, patch_dim)

        gauss_window = gaussian_window( # h', w'
            n=2*self.subsample_factor,
            var=self.subsample_factor**2,
            dtype=act_maps.dtype,
            device=device
        )

        output_maps = F.conv2d(
            act_maps,  # batch * p_k, 1, h, w
            gauss_window.view(1, 1, *gauss_window.shape),  # 1, 1, h', w'
            stride=self.subsample_factor,
            padding=self.subsample_factor//2
        )
        output_maps = output_maps.view(
            n,
            self.out_filters,
            *output_maps.shape[2:]
        ).permute(0, 2, 3, 1).contiguous()

        return output_maps # batch, p_k, h", w"

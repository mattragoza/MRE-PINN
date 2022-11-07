import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .generic import get_activ_fn, ParallelNet


class FNO(torch.nn.Module):

    def __init__(
        self,
        n_spatial_dims,
        n_channels_in,
        n_channels_out,
        n_spatial_freqs,
        n_channels_model,
        n_spectral_convs,
        activ_fn,
        omega,
        device='cuda'
    ):
        super().__init__()

        self.fc_in = nn.Linear(n_channels_in, n_channels_model)

        self.convs = []
        for i in range(n_spectral_blocks):
            block = SpectralConv(
                n_spatial_dims=n_spatial_dims,
                n_channels_in=n_channels_model,
                n_channels_out=n_channels_model,
                n_spatial_freqs=n_spatial_freqs
            )
            self.blocks.append(block)
            self.add_module(f'conv{i+1}', block)

        self.fc_out = nn.Linear(n_channels_model, n_channels_out)
        self.activ_fn = get_activ_fn(activ_fn)
        self.regularizer = None

    def forward(self, inputs):
        '''
        Args:
            a: (batch_size, n_x, n_y, n_z, n_channels_in)
            x: (batch_size, n_x, n_y, n_z, n_spatial_dims)
            y: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            u: (batch_size, n_x, n_y, n_z, n_channels_out)
        '''
        a, x, y = inputs
        h = self.fc_in(a)
        for i, conv in enumerate(self.convs):
            if i < len(self.convs) - 1:
                h = self.activ_fn(conv(h, x, x))
            else:
                h = self.activ_fn(conv(h, x, y))
        u = self.fc_out(h)
        return u


class SpectralConv(torch.nn.Module):

    def __init__(
        self,
        n_spatial_dims,
        n_channels_in,
        n_channels_out,
    ):
        pass

    def forward(self, h, x, y):
        '''
        Args:
            h: (batch_size, n_x, n_y, n_z, n_channels_in)
            x: (batch_size, n_x, n_y, n_z, n_spatial_dims)
            y: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            g: (batch_size, n_x, n_y, n_z, n_channels_out)
        '''
        H = torch.fft.fftn(H, dim=[1,2,3])
      

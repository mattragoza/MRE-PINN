import numpy as np
import torch
from torch import nn

from .generic import get_activ_fn, ParallelNet


def xavier(fan_in, fan_out, gain=1, c=6):
    scale = gain * np.sqrt(c / (fan_in + fan_out))
    w = torch.rand(fan_in, fan_out, device='cuda', dtype=torch.float32)
    return nn.Parameter((2 * w - 1) * scale)


class SpectralTransformer(torch.nn.Module):

    def __init__(
        self,
        n_spatial_dims,
        n_channels_in,
        n_channels_out,
        n_spatial_freqs,
        n_channels_model,
        n_blocks,
        activ_fn,
        omega,
        device='cuda'
    ):
        super().__init__()

        self.spectral_fwd = SpectralTransform(
            n_spatial_dims=n_spatial_dims,
            n_channels_in=n_channels_in,
            n_spatial_freqs=n_spatial_freqs,
            n_channels_out=n_channels_model,
            omega=omega,
        )

        self.blocks = []
        for i in range(n_blocks):
            block = SpectralBlock(
                n_channels=n_channels_model,
                activ_fn=activ_fn,
                device=device
            )
            self.blocks.append(block)
            self.add_module(f'block{i}', block)

        self.spectral_inv = SpectralInverse(
            n_spatial_dims=n_spatial_dims,
            n_channels_in=n_channels_model,
            n_spatial_freqs=n_spatial_freqs,
            n_channels_out=n_channels_out,
            omega=omega
        )
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
        h = self.spectral_fwd(a, x)
        for block in self.blocks:
            h = block(h)
        u = self.spectral_inv(h, y)
        return u


class SpectralTransform(torch.nn.Module):

    def __init__(
        self,
        n_spatial_dims,
        n_channels_in,
        n_channels_out,
        n_spatial_freqs,
        omega,
    ):
        super().__init__()
        self.a_weights = xavier(n_channels_in, n_channels_out)
        self.x_weights = xavier(n_spatial_dims, n_spatial_freqs, omega)

    def forward(self, a, x):
        '''
        Args:
            a: (batch_size, n_x, n_y, n_z, n_channels_in)
            x: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            h: (batch_size, n_spatial_freqs, n_channels_out)
        '''
        A = torch.einsum('bxyzi,im->bxyzm', a, self.a_weights)
        X = torch.einsum('bxyzd,df->bxyzf', x, self.x_weights)
        X = torch.exp(-2j * np.pi * X)
        n_xyz = x.shape[1] * x.shape[2] * x.shape[3]
        return torch.einsum('bxyzm,bxyzf->bfm', A + 0j, X) / np.sqrt(n_xyz)


class SpectralInverse(torch.nn.Module):

    def __init__(
        self,
        n_spatial_dims,
        n_channels_in,
        n_channels_out,
        n_spatial_freqs,
        omega,
    ):
        super().__init__()
        self.y_weights = xavier(n_spatial_dims, n_spatial_freqs, omega)
        self.u_weights = xavier(n_channels_in, n_channels_out)

    def forward(self, h, y):
        '''
        Args:
            h: (batch_size, n_spatial_freqs, n_channels_in)
            y: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            u: (batch_size, n_x, n_y, n_z, n_channels_out)
        '''
        Y = torch.einsum('bxyzd,df->bxyzf', y, self.y_weights)
        Y = torch.exp(2j * np.pi * Y)
        n_xyz = y.shape[1] * y.shape[2] * y.shape[3]
        U = torch.einsum('bfm,bxyzf->bxyzm', h, Y).real / np.sqrt(n_xyz)
        return torch.einsum('bxyzm,mo->bxyzo', U, self.u_weights)


class SpectralBlock(torch.nn.Module):

    def __init__(
        self,
        n_channels,
        activ_fn,
        device='cuda'
    ):
        super().__init__()
        self.fc = torch.nn.Linear(
            n_channels, n_channels, device=device, dtype=torch.complex64
        )
        self.activ_fn = get_activ_fn(activ_fn)
        self.regularizer = None

    def forward(self, h):
        '''
        Args:
            h: (batch_size, n_spatial_freqs, n_channels)
        Returns:
            out: (batch_size, n_spatial_freqs, n_channels)
        '''
        h = self.fc(h) + h
        h_real = self.activ_fn(h.real)
        h_imag = self.activ_fn(h.imag)
        return torch.complex(h_real, h_imag)

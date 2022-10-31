import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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
        n_hidden_layers,
        n_spectral_blocks,
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
            n_hidden_layers=n_hidden_layers,
            activ_fn=activ_fn,
            omega=omega,
        )

        self.blocks = []
        for i in range(n_spectral_blocks):
            block = SpectralBlock(
                n_channels=n_channels_model,
                activ_fn=activ_fn,
                device=device
            )
            self.blocks.append(block)
            self.add_module(f'spectral_block{i+1}', block)

        self.spectral_inv = SpectralInverse(
            n_spatial_dims=n_spatial_dims,
            n_channels_in=n_channels_model,
            n_spatial_freqs=n_spatial_freqs,
            n_channels_out=n_channels_out,
            n_hidden_layers=n_hidden_layers,
            activ_fn=activ_fn,
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
        n_hidden_layers,
        activ_fn,
        omega,
    ):
        super().__init__()
        self.a_linear = nn.Linear(n_channels_in, n_channels_out)

        self.linears = []
        for i in range(n_hidden_layers):
            linear = nn.Linear(n_channels_out, n_channels_out)
            self.add_module(f'A_linear{i+1}', linear)
            self.linears.append(linear)

        self.x_linear = nn.Linear(n_spatial_dims, n_spatial_freqs)
        with torch.no_grad():
            self.x_linear.weight *= omega

        self.activ_fn = get_activ_fn(activ_fn)

    def forward(self, a, x):
        '''
        Args:
            a: (batch_size, n_x, n_y, n_z, n_channels_in)
            x: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            h: (batch_size, n_spatial_freqs, n_channels_out)
        '''
        A = self.activ_fn(self.a_linear(a))

        for linear in self.linears:
            A = self.activ_fn(linear(A))

        X = self.x_linear(x)
        X = torch.exp(-2j * np.pi * X)

        n_xyz = x.shape[1] * x.shape[2] * x.shape[3]
        return torch.einsum('bxyzm,bxyzf->bfm', A + 0j, X) #/ n_xyz


class SpectralInverse(torch.nn.Module):

    def __init__(
        self,
        n_spatial_dims,
        n_channels_in,
        n_channels_out,
        n_spatial_freqs,
        n_hidden_layers,
        activ_fn,
        omega,
    ):
        super().__init__()
        self.y_linear = nn.Linear(n_spatial_dims, n_spatial_freqs)
        with torch.no_grad():
            self.y_linear.weight *= omega

        self.linears = []
        for i in range(n_hidden_layers):
            linear = nn.Linear(n_channels_in, n_channels_in)
            self.add_module(f'U_linear{i+1}', linear)
            self.linears.append(linear)
        self.activ_fn = get_activ_fn(activ_fn)

        self.u_linear = nn.Linear(n_channels_in, n_channels_out)

    def forward(self, h, y):
        '''
        Args:
            h: (batch_size, n_spatial_freqs, n_channels_in)
            y: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            u: (batch_size, n_x, n_y, n_z, n_channels_out)
        '''
        Y = self.y_linear(y)
        Y = torch.exp(2j * np.pi * Y)

        n_xyz = y.shape[1] * y.shape[2] * y.shape[3]
        U = torch.einsum('bfm,bxyzf->bxyzm', h, Y).real / n_xyz
        U = self.activ_fn(U)

        for linear in self.linears:
            U = self.activ_fn(linear(U))

        return self.u_linear(U)


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

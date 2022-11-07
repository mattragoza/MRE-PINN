import numpy as np
import torch
#torch.backends.cudnn.enabled = False
from torch import nn
from torch.nn import functional as F

from .generic import get_activ_fn, ParallelNet


def xavier(fan_in, fan_out, gain=1, c=6):
    scale = gain * np.sqrt(c / (fan_in + fan_out))
    w = torch.rand(fan_in, fan_out, device='cuda', dtype=torch.float32)
    return nn.Parameter((2 * w - 1) * scale)


class SpectralOperator(torch.nn.Module):

    def __init__(
        self,
        n_spatial_dims,
        n_channels_in,
        n_channels_out,
        n_spatial_freqs,
        n_channels_conv,
        n_conv_blocks,
        n_conv_per_block,
        activ_fn,
        omega,
        device='cuda'
    ):
        super().__init__()

        self.a_net = SpectralEncoder(
            n_spatial_dims=n_spatial_dims,
            n_channels_in=n_channels_in,
            n_spatial_freqs=n_spatial_freqs,
            n_channels_out=n_channels_conv,
            n_conv_blocks=n_conv_blocks,
            n_conv_per_block=n_conv_per_block,
            activ_fn=activ_fn,
            omega=omega
        )

        self.u_net = SpectralDecoder(
            n_spatial_dims=n_spatial_dims,
            n_channels_in=n_channels_conv,
            n_spatial_freqs=n_spatial_freqs,
            n_channels_out=n_channels_out,
            activ_fn=activ_fn,
            omega=omega
        )

        self.mu_net = SpectralDecoder(
            n_spatial_dims=n_spatial_dims,
            n_channels_in=n_channels_conv,
            n_spatial_freqs=n_spatial_freqs,
            n_channels_out=n_channels_out,
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
        h = self.a_net(a, x)
        u = self.u_net(h, y)
        mu = self.mu_net(h, y)
        return u, mu


class SpectralEncoder(torch.nn.Module):

    def __init__(
        self,
        n_spatial_dims,
        n_channels_in,
        n_spatial_freqs,
        n_channels_out,
        n_conv_blocks,
        n_conv_per_block,
        activ_fn,
        omega
    ):
        super().__init__()
        self.a_linear = nn.Linear(n_channels_in, n_channels_out)

        self.blocks = []
        for i in range(n_conv_blocks):
            block = ConvBlock(
                n_channels_in=n_channels_out,
                n_channels_out=n_channels_out,
                n_conv_layers=n_conv_per_block,
                activ_fn=activ_fn,
            )
            self.add_module(f'conv_block{i+1}', block)
            self.blocks.append(block)

        self.x_linear = nn.Linear(n_spatial_dims, n_spatial_freqs)
        self.omega = omega

    def forward(self, a, x):
        '''
        Args:
            a: (batch_size, n_x, n_y, n_z, n_channels_in)
            x: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            h: (batch_size, n_spatial_freqs, n_channels_out)
        '''
        A = self.a_linear(a)

        A = torch.permute(A, (0, 4, 1, 2, 3))
        x = torch.permute(x, (0, 4, 1, 2, 3))
        for block in self.blocks:
            A = block.forward(A)
            x = block.pool(x)
        A = torch.permute(A, (0, 2, 3, 4, 1))
        x = torch.permute(x, (0, 2, 3, 4, 1))

        X = self.x_linear(x)
        X = torch.sin(-2 * np.pi * self.omega * X)

        n_xyz = x.shape[1] * x.shape[2] * x.shape[3]
        return torch.einsum('bxyzf,bxyzc->bfc', X, A)


class ConvBlock(torch.nn.Module):

    def __init__(self, n_channels_in, n_channels_out, n_conv_layers, activ_fn):
        super().__init__()
        self.convs = []
        for i in range(n_conv_layers):
            conv = nn.Conv3d(n_channels_in, n_channels_out, kernel_size=3, padding=1)
            self.convs.append(conv)
            self.add_module(f'conv{i}', conv)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.activ_fn = get_activ_fn(activ_fn)

    def forward(self, h):
        for conv in self.convs:
            h = self.activ_fn(conv(h))
        return self.pool(h)


class SpectralDecoder(torch.nn.Module):

    def __init__(
        self,
        n_spatial_dims,
        n_channels_in,
        n_spatial_freqs,
        n_channels_out,
        activ_fn,
        omega,
    ):
        super().__init__()
        self.y_linear = nn.Linear(n_spatial_dims, n_spatial_freqs)
        self.omega = omega

        self.h_linear = nn.Linear(n_channels_in, n_spatial_freqs)

        self.u_linear = nn.Linear(n_spatial_freqs, n_channels_out)
        self.activ_fn = get_activ_fn(activ_fn)

    def forward(self, h, y):
        '''
        Args:
            h: (batch_size, n_spatial_freqs, n_channels_in)
            y: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            u: (batch_size, n_x, n_y, n_z, n_channels_out)
        '''
        Y = self.y_linear(y)
        Y = torch.sin(2 * np.pi * self.omega * Y)

        H = self.h_linear(h)
        H = self.activ_fn(H)

        U = torch.einsum('bfc,bxyzf->bxyzc', H, Y)
        U = U / (y.shape[1] * y.shape[2] * y.shape[3])
        U = self.activ_fn(U)

        return self.u_linear(U)

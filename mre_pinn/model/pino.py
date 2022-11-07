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


class HyperPINN(torch.nn.Module):

    def __init__(
        self,
        n_channels_in,
        n_channels_out,
        n_channels_conv,
        n_conv_blocks,
        n_conv_per_block,
        activ_fn,
        n_latent,
        n_spatial_dims,
        n_spatial_freqs,
        omega,
        device='cuda'
    ):
        super().__init__()

        self.a_net = CNN(
            n_channels_in=n_channels_in,
            n_channels_conv=n_channels_conv,
            n_conv_blocks=n_conv_blocks,
            n_conv_per_block=n_conv_per_block,
            activ_fn=activ_fn,
            n_latent=n_latent
        )

        self.u_net = HypoPINN(
            n_latent=n_latent,
            n_spatial_dims=n_spatial_dims,
            n_spatial_freqs=n_spatial_freqs,
            n_channels_out=n_channels_out,
            activ_fn=activ_fn,
            omega=omega
        )

        self.mu_net = HypoPINN(
            n_latent=n_latent,
            n_spatial_dims=n_spatial_dims,
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
            mask
        Returns:
            u: (batch_size, n_x, n_y, n_z, n_channels_out)
        '''
        a, x, y, mask = inputs
        h = self.a_net(a)
        u = self.u_net(h, y)
        mu = self.mu_net(h, y)
        return u, mu


class CNN(torch.nn.Module):

    def __init__(
        self,
        n_channels_in,
        n_channels_conv,
        n_conv_blocks,
        n_conv_per_block,
        activ_fn,
        n_latent
    ):
        super().__init__()
        self.embed = nn.Linear(n_channels_in, n_channels_conv)

        self.blocks = []
        for i in range(n_conv_blocks):
            block = ConvBlock(
                n_channels_in=n_channels_conv,
                n_channels_out=n_channels_conv,
                n_conv_layers=n_conv_per_block,
                activ_fn=activ_fn,
            )
            self.add_module(f'conv_block{i+1}', block)
            self.blocks.append(block)

        self.latent = nn.Linear(n_channels_conv, n_latent)
        self.activ_fn = get_activ_fn(activ_fn)

    def forward(self, a):
        '''
        Args:
            a: (batch_size, n_x, n_y, n_z, n_channels_in)
        Returns:
            h: (batch_size, n_latent)
        '''
        A = self.embed(a)
        A = torch.permute(A, (0, 4, 1, 2, 3))
        for block in self.blocks:
            A = block.forward(A)
        A = A.mean(dim=(2,3,4))
        return self.activ_fn(self.latent(A))


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


class HypoPINN(torch.nn.Module):

    def __init__(
        self,
        n_latent,
        n_spatial_dims,
        n_spatial_freqs,
        n_channels_out,
        activ_fn,
        omega,
    ):
        super().__init__()

        #self.x_linear = HypoLinear(n_latent, n_spatial_dims, n_spatial_freqs, activ_fn)
        self.x_linear = nn.Linear(n_spatial_dims, n_spatial_freqs)
        self.omega = omega

        #self.U_linear = HypoLinear(n_latent, n_spatial_freqs, n_spatial_freqs, activ_fn)
        self.U_linear = nn.Linear(n_spatial_freqs, n_spatial_freqs)
        self.activ_fn = get_activ_fn(activ_fn)

        #self.u_linear = HypoLinear(n_latent, n_spatial_freqs, n_channels_out, activ_fn)
        self.u_linear = nn.Linear(n_spatial_freqs, n_channels_out)

    def forward(self, h, x):
        '''
        Args:
            h: (batch_size, n_latent)
            x: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            u: (batch_size, n_x, n_y, n_z, n_channels_out)
        '''
        X = self.x_linear(x)
        X = torch.sin(2 * np.pi * self.omega * X)

        U = self.U_linear(X)
        U = self.activ_fn(U)

        return self.u_linear(U)


class HypoLinear(torch.nn.Module):

    def __init__(self, n_latent, n_channels_in, n_channels_out, activ_fn):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.w_linear = nn.Linear(n_latent, n_channels_in * n_channels_out)
        self.b_linear = nn.Linear(n_latent, n_channels_out)
        self.activ_fn = get_activ_fn(activ_fn)

    def forward(self, latent, x):
        '''
        Args:
            latent: (batch_size, n_latent)
            input: (batch_size, n_x, n_y, n_z, n_channels_in)
        Returns:
            output: (batch_size, n_x, n_y, n_z, n_channels_out)
        '''
        w = self.w_linear(latent).reshape(-1, self.n_channels_in, self.n_channels_out)
        w = self.activ_fn(w)
        b = self.b_linear(latent).reshape(-1, 1, 1, 1, self.n_channels_out)
        b = self.activ_fn(b)
        return torch.einsum('bio,bxyzi->bxyzo', w, x) + b

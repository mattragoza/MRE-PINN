import numpy as np
import torch
from torch import nn

from .generic import get_activ_fn, ParallelNet


class PINO(torch.nn.ModuleList):
    '''
    Physics-informed neural operator.
    '''
    def __init__(
        self,
        n_spatial_dims,
        n_channels_in,
        n_blocks,
        n_hidden,
        n_modes,
        n_channels_out,
        activ_fn
    ):
        super().__init__()

        self.fc_in = torch.nn.Linear(n_channels_in, n_hidden)

        self.blocks = []
        for i in range(n_blocks):
            block = PINOBlock(
                n_spatial_dims=n_spatial_dims,
                n_channels_in=n_hidden,
                n_channels_out=n_hidden,
                n_modes=n_modes,
                activ_fn=activ_fn
            )
            self.blocks.append(block)
            self.add_module(f'block{i}', block)

        self.conv_out = SpectralConv3d(
            n_spatial_dims, n_hidden, n_channels_out, n_modes
        )
        self.regularizer = None

    def forward(self, inputs):
        '''
        Args:
            inputs: Tuple of input tensors:
                a: (batch_size, n_x, n_y, n_z, n_channels_in) tensor
                x: (batch_size, n_x, n_y, n_z, n_spatial_dims) tensor
                y: (batch_size, n_x, n_y, n_z, n_spatial_dims) tensor
        Returns:
            u: (batch_size, n_x, n_y, n_z, n_channels_out) output tensor
        '''
        a, x, y = inputs
        h = self.fc_in(a)
        for block in self.blocks:
            h = block(h, x)
        u = self.conv_out(h, x, y)
        return u


class PINOBlock(torch.nn.Module):

    def __init__(
        self, n_spatial_dims, n_channels_in, n_channels_out, n_modes, activ_fn
    ):
        super().__init__()
        self.conv = SpectralConv3d(
            n_spatial_dims, n_channels_in, n_channels_out, n_modes
        )
        self.fc = torch.nn.Linear(n_channels_in, n_channels_out)
        self.activ_fn = get_activ_fn(activ_fn)

    def forward(self, h, x):
        '''
        Args:
            h: (batch_size, n_x, n_y, n_z, n_channels_in)
            x: (batch_size, n_x, n_y, n_z, n_spatial_dims)
            y: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            out: (batch_size, n_x, n_y, n_z, n_channels_out)
        '''
        h = self.conv(h, x, x) + self.fc(x)
        return self.activ_fn(h)


class SpectralConv3d(torch.nn.Module):

    def __init__(self, n_spatial_dims, n_channels_in, n_channels_out, n_modes=16):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.n_modes = n_modes

        scale = 1e-3
        self.modes = nn.Parameter(scale * torch.rand(
            (n_modes, n_spatial_dims), dtype=torch.float32
        ))
        scale = 1 / (n_channels_in * n_channels_out)
        self.kernel = nn.Parameter(scale * torch.rand(
            (n_modes, n_channels_in, n_channels_out), dtype=torch.complex64
        ))

    def __repr__(self):
        return f'SpectralConv3d(n_spatial_dims={self.n_spatial_dims}, n_channels_in={self.n_channels_in}, n_channels_out={self.n_channels_out}, n_modes={self.n_modes})'

    def forward(self, h, x, y):
        '''
        Args:
            h: (batch_size, n_x, n_y, n_z, n_channels_in)
            x: (batch_size, n_x, n_y, n_z, n_spatial_dims)
            y: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            g: (batch_size, n_x, n_y, n_z, n_channels_in)
        '''
        assert h.shape[-1] == self.n_channels_in
        assert x.shape[-1] == self.n_spatial_dims
        assert y.shape[-1] == self.n_spatial_dims

        batch_size = h.shape[0]
        spatial_dims = (1, 2, 3)
        n_x, n_y, n_z = h.shape[1:4]

        # convert to frequency domain
        x_dot_s = torch.einsum('bxyzd,fd->bxyzf', x, self.modes)
        phi_x = torch.exp(-2j * np.pi * x_dot_s)
        H = torch.einsum('bxyzi,bxyzf->bfi', h + 0j, phi_x)

        # frequency-domain convolution
        G = torch.einsum('bfi,fio->bfo', H, self.kernel)

        # convert back to spatial domain
        y_dot_s = torch.einsum('bxyzd,fd->bxyzf', y, self.modes)
        phi_y = torch.exp(2j * np.pi * y_dot_s)
        g = torch.einsum('bfo,bxyzf->bxyzo', G, phi_y).real
        print(g.shape)

        return g

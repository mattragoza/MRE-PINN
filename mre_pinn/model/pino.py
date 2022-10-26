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
        activ_fn,
        device='cuda'
    ):
        super().__init__()

        self.fc_in = torch.nn.Linear(n_channels_in, n_hidden, device=device)

        self.blocks = []
        for i in range(n_blocks):
            block = PINOBlock(
                n_spatial_dims=n_spatial_dims,
                n_channels_in=n_hidden,
                n_channels_out=n_hidden,
                n_modes=n_modes,
                activ_fn=activ_fn,
                device=device
            )
            self.blocks.append(block)
            self.add_module(f'block{i}', block)

        self.conv_out = SpectralConv3d(
            n_spatial_dims, n_hidden, n_channels_out, n_modes, device=device
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
        self,
        n_spatial_dims,
        n_channels_in,
        n_channels_out,
        n_modes,
        activ_fn,
        device='cuda'
    ):
        super().__init__()
        self.att = SpectralAttention(
            n_spatial_dims,
            n_channels_in,
            n_channels_out,
            n_modes,
            device=device
        )
        self.fc = torch.nn.Linear(n_channels_in, n_channels_out, device=device)
        self.activ_fn = get_activ_fn(activ_fn)

    def forward(self, a, x, y):
        '''
        Args:
            a: (batch_size, n_x, n_y, n_z, n_channels_in)
            x: (batch_size, n_x, n_y, n_z, n_spatial_dims)
            y: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            out: (batch_size, n_x, n_y, n_z, n_channels_out)
        '''
        h = self.att(h, x, x) + self.fc(h)
        return self.activ_fn(h)


class SpectralAttention(torch.nn.Module):

    def __init__(
        self,
        n_spatial_dims,
        n_channels_in,
        n_channels_out,
        n_modes=16,
        device='cuda'
    ):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.n_modes = n_modes

        self.init_weights(device=device)
        self.regularizer = None

    def init_weights(self, omega=30/300, c=6, device='device'):

        scale = np.sqrt(c / (self.n_channels_in + self.n_modes))
        shape = (self.n_modes, self.n_channels_in)
        modes_a = 2 * torch.rand(shape, device=device, dtype=torch.float32) - 1
        self.modes_a = nn.Parameter(scale * modes_a)

        scale = np.sqrt(c / (self.n_spatial_dims + self.n_modes)) * omega
        shape = (self.n_modes, self.n_spatial_dims)
        modes_x = 2 * torch.rand(shape, device=device, dtype=torch.float32) - 1
        self.modes_x = nn.Parameter(scale * modes_x)

        scale = np.sqrt(c / (self.n_spatial_dims + self.n_modes)) * omega
        shape = (self.n_modes, self.n_spatial_dims)
        modes_y = 2 * torch.rand(shape, device=device, dtype=torch.float32) - 1
        self.modes_y = nn.Parameter(scale * modes_y)

        scale = np.sqrt(c / (self.n_modes + self.n_channels_out))
        shape = (self.n_modes, self.n_channels_out)
        modes_u = 2 * torch.rand(shape, device=device, dtype=torch.float32) - 1
        self.modes_u = nn.Parameter(scale * modes_u)

    def __repr__(self):
        return  (f'SpectralAttention('
            f'n_spatial_dims={self.n_spatial_dims}, '
            f'n_channels_in={self.n_channels_in}, '
            f'n_channels_out={self.n_channels_out}, '
            f'n_modes={self.n_modes}' 
        ')')

    def forward(self, inputs):
        '''
        Args:
            a: (batch_size, n_x, n_y, n_z, n_channels_in)
            x: (batch_size, n_x, n_y, n_z, n_spatial_dims)
            y: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            u: (batch_size, n_x, n_y, n_z, n_channels_in)
        '''    
        a, x, y = inputs
        A = torch.einsum('bxyzi,fi->bxyzf', a, self.modes_a)
        X = torch.einsum('bxyzd,gd->bxyzg', x, self.modes_x)
        Y = torch.einsum('bxyzd,gd->bxyzg', y, self.modes_y)

        X = torch.exp(-2j * np.pi * X)
        Y = torch.exp( 2j * np.pi * Y)

        N = y.shape[1] * y.shape[2] * y.shape[3]
        AX = torch.einsum('bxyzf,bxyzg->bfg', A + 0j, X)
        U = torch.einsum('bfg,bxyzg->bxyzf', AX, Y).real / N

        u = torch.einsum('bxyzf,fo->bxyzo', U, self.modes_u)
        return u

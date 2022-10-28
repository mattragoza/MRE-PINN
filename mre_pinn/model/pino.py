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
        n_channels_out,
        n_blocks,
        n_modes,
        activ_fn,
        device='cuda'
    ):
        super().__init__()
        self.spectral = SpectralTransform(
            n_spatial_dims=n_spatial_dims,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            n_modes=n_modes,
        )
        self.blocks = []
        for i in range(n_blocks):
            block = PINOBlock(
                n_channels=n_modes,
                activ_fn=activ_fn,
                device=device
            )
            self.blocks.append(block)
            self.add_module(f'block{i}', block)

        self.regularizer = None

    def forward(self, inputs):
        '''
        Args:
            inputs: Tuple of input tensors:
                a: (batch_size, n_x, n_y, n_z, n_channels_in) tensor
                x: (batch_size, n_x, n_y, n_z, n_spatial_dims) tensor
                y: (batch_size, m_x, m_y, m_z, n_spatial_dims) tensor
        Returns:
            u: (batch_size, m_x, m_y, m_z, n_channels_out) output tensor
        '''
        a, x, y = inputs
        h = self.spectral.forward(a, x)
        for block in self.blocks:
            h = block(h)
        u = self.spectral.inverse(h, y)
        return u


class PINOBlock(torch.nn.Module):

    def __init__(
        self,
        n_channels,
        activ_fn,
        device='cuda'
    ):
        super().__init__()
        self.fc = torch.nn.Linear(n_channels, n_channels, device=device)
        self.activ_fn = get_activ_fn(activ_fn)
        self.regularizer = None

    def forward(self, h):
        '''
        Args:
            h: (batch_size, n_modes, n_modes)
        Returns:
            out: (batch_size, n_modes, n_modes)
        '''
        return self.fc()


class SpectralTransform(torch.nn.Module):

    def __init__(
        self,
        n_spatial_dims,
        n_channels_in,
        n_channels_out,
        n_modes=128,
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

        a_scale = np.sqrt(c / (self.n_channels_in + self.n_modes))
        a_shape = (self.n_modes, self.n_channels_in)
        a_modes = 2 * torch.rand(a_shape, device=device, dtype=torch.float32) - 1
        self.a_modes = nn.Parameter(a_scale * a_modes)

        x_scale = np.sqrt(c / (self.n_spatial_dims + self.n_modes)) * omega
        x_shape = (self.n_modes, self.n_spatial_dims)
        x_modes = 2 * torch.rand(x_shape, device=device, dtype=torch.float32) - 1
        self.x_modes = nn.Parameter(x_scale * x_modes)

        y_scale = np.sqrt(c / (self.n_spatial_dims + self.n_modes)) * omega
        y_shape = (self.n_modes, self.n_spatial_dims)
        y_modes = 2 * torch.rand(y_shape, device=device, dtype=torch.float32) - 1
        self.y_modes = nn.Parameter(y_scale * y_modes)

        u_scale = np.sqrt(c / (self.n_modes + self.n_channels_out))
        u_shape = (self.n_modes, self.n_channels_out)
        u_modes = 2 * torch.rand(u_shape, device=device, dtype=torch.float32) - 1
        self.u_modes = nn.Parameter(u_scale * u_modes)

    def __repr__(self):
        return  (
            f'{type(self).__name__}('
                f'n_spatial_dims={self.n_spatial_dims}, '
                f'n_channels_in={self.n_channels_in}, '
                f'n_channels_out={self.n_channels_out}, '
                f'n_modes={self.n_modes}'
            ')'
        )

    def forward(self, a, x):
        '''
        Args:
            a: (batch_size, n_x, n_y, n_z, n_channels_in)
            x: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            h: (batch_size, n_modes, n_modes)
        '''
        A = torch.einsum('bxyzi,mi->bxyzm', a, self.a_modes)
        X = torch.einsum('bxyzd,sd->bxyzs', x, self.x_modes)
        X = torch.exp(-2j * np.pi * X)
        n_xyz = x.shape[1] * x.shape[2] * x.shape[3]
        return torch.einsum('bxyzm,bxyzs->bms', A + 0j, X) / np.sqrt(n_xyz)

    def inverse(self, h, y):
        '''
        Args:
            h: (batch_size, n_modes, n_modes)
            y: (batch_size, n_x, n_y, n_z, n_spatial_dims)
        Returns:
            u: (batch_size, n_x, n_y, n_z, n_channels_out)
        '''
        Y = torch.einsum('bxyzd,sd->bxyzs', y, self.y_modes)
        Y = torch.exp(-2j * np.pi * Y)
        n_xyz = y.shape[1] * y.shape[2] * y.shape[3]
        U = torch.einsum('bms,bxyzs->bxyzm', h, Y).real / np.sqrt(n_xyz)
        return torch.einsum('bxyzm,mo->bxyzo', U, self.u_modes)

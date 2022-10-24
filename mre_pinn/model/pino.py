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

        self.fc_input = torch.nn.Linear(n_spatial_dims + n_channels_in, n_hidden)

        self.blocks = []
        for i in range(n_blocks):
            block = PINOBlock(
                n_channels_in=n_hidden,
                n_channels_out=n_hidden,
                n_modes=n_modes,
                activ_fn=activ_fn
            )
            self.blocks.append(block)
            self.add_module(f'block{i}', block)

        self.fc_output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, a, x):
        '''
        Args:
            a: (batch_size, n_x, n_y, n_z, n_channels_in) tensor
            x: (batch_size, n_x, n_y, n_z, n_spatial_dims) tensor
        Returns:
            u: (batch_size, n_x, n_y, n_z, n_channels_out) tensor
        '''
        h = self.fc_input(torch.cat([a, x], dim=-1))
        for block in self.blocks:
            h = block(h)
        u = self.fc_output(h)
        return u


class PINOBlock(torch.nn.Module):

    def __init__(self, n_channels_in, n_channels_out, n_modes, activ_fn):
        super().__init__()
        self.conv = SpectralConv3d(n_channels_in, n_channels_out, n_modes)
        self.fc = torch.nn.Linear(n_channels_in, n_channels_out)
        self.activ_fn = get_activ_fn(activ_fn)

    def forward(self, h):
        '''
        Args:
            h: (batch_size, n_channels_in, n_x, n_y, n_z)
        Returns:
            out: (batch_size, n_channels_out, n_x, n_y, n_z)
        '''
        return self.activ_fn(self.conv(h) + self.fc(h))


class SpectralConv3d(torch.nn.Module):

    def __init__(self, n_channels_in, n_channels_out, n_modes=16):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.n_modes = n_modes

        scale = 1 / (n_channels_in * n_channels_out)
        self.weights = nn.Parameter(scale * torch.rand(
            n_modes, n_modes, n_modes, n_channels_in, n_channels_out,
            dtype=torch.complex64
        ))

    def forward(self, h):
        '''
        Args:
            h: (batch_size, n_x, n_y, n_z, n_channels_in)
        Returns:
            out: (batch_size, n_x, n_y, n_z, n_channels_out)
        '''
        assert h.shape[-1] == self.n_channels_in
        batch_size = h.shape[0]
        spatial_dims = (1, 2, 3)
        n_x, n_y, n_z = h.shape[1:4]

        F_h = torch.fft.rfftn(h, dim=spatial_dims)
        n = self.n_modes
        F_out = torch.zeros(batch_size, n_x, n_y, n_z, self.n_channels_out)
        F_out[:,:n,:n,:n,:] = torch.einsum(
            'bxyzi,xyzio->bxyzo', F_h[:,:n,:n,:n,:], self.weights
        )
        return torch.fft.irfftn(F_out, s=(n_x, n_y, n_z), dim=spatial_dims)


class FFT(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, a, x):
        ctx.save_for_backward(a, x)
        return torch.fft.fftn(a)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

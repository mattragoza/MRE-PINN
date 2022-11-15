import time
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


def printt(**kwargs):
    '''
    Print tensor info.
    '''
    for key, val in kwargs.items():
        print(f'{key}\t{val.shape}\t{val.dtype}')


def describe(**kwargs):
    for key, val in kwargs.items():
        print(f'{key}\t{val.mean()}\t{val.std()}')



class HyperCNN(torch.nn.Module):

    def __init__(
        self,
        n_channels_in,
        n_channels_block,
        n_conv_per_block,
        n_conv_blocks,
        activ_fn,
        n_latent,
        n_pinn_layers,
        n_pinn_hidden,
        u_omega,
        u_scale,
        u_loc,
        mu_omega,
        mu_scale,
        mu_loc,
        width_factor=1,
        width_term=0,
        depth_factor=1,
        depth_term=0,
        skip_connect=True,
        dense=True,
        parallel=False,
        debug=False
    ):
        super().__init__()

        self.u_cnn = CNN(
            n_channels_in=n_channels_in,
            n_channels_block=n_channels_block,
            n_conv_per_block=n_conv_per_block,
            n_conv_blocks=n_conv_blocks,
            activ_fn=activ_fn,
            n_output=n_latent,
            width_factor=width_factor,
            width_term=width_term,
            depth_factor=depth_factor,
            depth_term=depth_term,
            skip_connect=skip_connect,
            debug=debug
        )
        self.u_norm = nn.LayerNorm(n_latent)

        if parallel:
            self.mu_cnn = CNN(
                n_channels_in=n_channels_in,
                n_channels_block=n_channels_block,
                n_conv_per_block=n_conv_per_block,
                n_conv_blocks=n_conv_blocks,
                activ_fn=activ_fn,
                n_output=n_latent,
                width_factor=width_factor,
                width_term=width_term,
                depth_factor=depth_factor,
                depth_term=depth_term,
                skip_connect=skip_connect,
                debug=debug
            )
            self.mu_norm = nn.LayerNorm(n_latent)

        self.u_pinn = HyperPINN(
            n_latent=n_latent,
            n_layers=n_pinn_layers,
            n_hidden=n_pinn_hidden,
            omega=u_omega,
            scale=u_scale,
            loc=u_loc,
            dense=dense
        )
        self.mu_pinn = HyperPINN(
            n_latent=n_latent,
            n_layers=n_pinn_layers,
            n_hidden=n_pinn_hidden,
            omega=mu_omega,
            scale=mu_scale,
            loc=mu_loc,
            dense=dense
        )

        self.parallel = parallel
        self.regularizer = None

    def forward(self, inputs, debug=False):
        u, x = inputs
        h = self.u_cnn(u)
        h = self.u_norm(h)
        h = torch.tanh(h)
        u_pred = self.u_pinn(h, x)

        if self.parallel:
            h = self.mu_cnn(u)
            h = self.mu_norm(h)
            h = torch.tanh(h)

        mu_pred = self.mu_pinn(h, x)
        mu_pred = torch.nn.functional.leaky_relu(mu_pred)

        return u_pred, mu_pred


class CNN(torch.nn.Module):
    '''
    Convolutional neural network.
    '''
    def __init__(
        self,
        n_channels_in,
        n_channels_block,
        n_conv_per_block,
        n_conv_blocks,
        activ_fn,
        n_output,
        width_factor=1,
        width_term=0,
        depth_factor=1,
        depth_term=0,
        skip_connect=True,
        debug=False
    ):
        super().__init__()
        xyz_shape = np.array([256, 256, 4])
        if debug:
            print('input\t\t', n_channels_in, xyz_shape)

        self.conv_in = nn.Conv3d(
            in_channels=n_channels_in,
            out_channels=n_channels_block,
            kernel_size=1
        )
        if debug:
            print('conv_in\t\t', n_channels_block, xyz_shape)

        n_channels_in = n_channels_block
        self.blocks = []
        self.pools = []
        for i in range(n_conv_blocks):

            block = ConvBlock(
                n_channels_in=n_channels_in,
                n_channels_block=n_channels_block,
                n_channels_out=n_channels_block,
                n_conv_layers=n_conv_per_block,
                activ_fn=activ_fn
            )
            self.blocks.append(block)
            self.add_module(f'conv_block{i}', block)

            if debug:
                print(f'conv_block{i}\t', n_channels_block, xyz_shape)

            if skip_connect:
                n_channels_in = n_channels_block + n_channels_in
            else:
                n_channels_in = n_channels_block

            pool_shape = np.where(xyz_shape > 1, 2, 1)
            if any(pool_shape > 1):
                pool_shape = tuple(pool_shape)
                pool = nn.AvgPool3d(kernel_size=pool_shape, stride=pool_shape)
                xyz_shape //= pool_shape
            else: # no pooling
                pool = nn.Identity()

            if debug:
                print(f'pool{i}\t\t', n_channels_block, xyz_shape)

            self.pools.append(pool)
            self.add_module(f'pool{i}', pool)

            n_channels_block = n_channels_block * width_factor + width_term
            n_conv_per_block = n_conv_per_block * depth_factor + depth_term

        self.linear_out = nn.Linear(
            in_features=n_channels_in * np.prod(xyz_shape),
            out_features=n_output
        )
        if debug:
            print(f'linear_out\t', n_output)
        self.skip_connect = skip_connect

    def forward(self, a, debug=False):
        '''
        Args:
            a: (batch_size, n_x, n_y, n_z, n_channels_in)
        Returns:
            h: (batch_size, n_output)
        '''
        a = torch.permute(a, (0, 4, 1, 2, 3))
        h = self.conv_in(a)

        for i, (block, pool) in enumerate(zip(self.blocks, self.pools)):
            g = block(h)
            if self.skip_connect:
                g = torch.cat([h, g], dim=1)
            h = pool(g)

        h = h.reshape(h.shape[0], -1)
        return self.linear_out(h)


class HyperLinear(torch.nn.Module):
    '''
    Linear layer with parameters that are
    conditioned on an input latent vector.
    '''
    def __init__(self, n_latent, n_features_in, n_features_out):
        super().__init__()
        self.linear_w = nn.Linear(n_latent, n_features_out * n_features_in)
        self.linear_b = nn.Linear(n_latent, n_features_out)

    def forward(self, h, x):
        '''
        Args:
            h: (batch_size, n_latent)
            x: (batch_size, n_x, n_y, n_z, n_features_in)
        Returns:
            y: (batch_size, n_x, n_y, n_z, n_features_out)
        '''
        w = self.linear_w(h)
        b = self.linear_b(h)
        n_features_out = b.shape[1]
        n_features_in = w.shape[1] // n_features_out
        w = w.view(-1, n_features_in, n_features_out)
        b = b.view(-1, 1, 1, 1, n_features_out)
        return torch.einsum('bxyzi,bio->bxyzo', x, w) + b


class HyperPINN(torch.nn.Module):

    def __init__(
        self,
        n_latent,
        n_layers,
        n_hidden,
        omega,
        scale,
        loc,
        dense=True
    ):
        assert n_layers > 0
        super().__init__()
        n_input = 6

        self.hiddens = []
        for i in range(n_layers - 1):
            hidden = HyperLinear(n_latent, n_input, n_hidden)
            if dense:
                n_input += n_hidden
            else:
                n_input = n_hidden
            self.hiddens.append(hidden)
            self.add_module(f'hidden{i}', hidden)

        self.output = HyperLinear(n_latent, n_input, 3)

        self.center = torch.zeros(1, 3, dtype=torch.float32)
        self.omega = torch.tensor(omega, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)
        self.loc = torch.tensor(loc, dtype=torch.float32)
        self.dense = dense

    def forward(self, h, x):
        '''
        Args:
            h: (batch_size, n_latent)
            x: (batch_size, n_x, n_y, n_z, 3)
        Returns:
            y: (batch_size, n_x, n_y, n_z, 1)
        '''
        x = x * self.omega + self.center

        # polar coordinates
        x, y, z = torch.split(x, 1, dim=-1)
        r = torch.sqrt(x**2 + y**2)
        eps = 1
        sin, cos = x / (r + eps), y / (r + eps)
        x = torch.cat([x, y, z, r, sin, cos], dim=-1)

        # hidden layers
        for i, hidden in enumerate(self.hiddens):
            y = torch.sin(hidden(h, x))
            if self.dense:
                x = torch.cat([x, y], dim=-1)
            else:
                x = y

        # output layer
        return self.output(h, x) * self.scale + self.loc


class UNet(torch.nn.Module):

    def __init__(
        self,
        n_channels_in,
        n_channels_out,
        n_channels_block,
        n_conv_per_block,
        n_unet_blocks,
        activ_fn,
        omega,
        alpha,
        skip_connect,
        width_factor=1,
        width_term=0,
        depth_factor=1,
        depth_term=0
    ):
        super().__init__()

        self.conv_in = nn.Conv3d(
            in_channels=n_channels_in,
            out_channels=n_channels_block,
            kernel_size=1
        )

        self.conv_block_in = ConvBlock(
            n_channels_in=n_channels_block,
            n_channels_block=n_channels_block,
            n_channels_out=n_channels_block,
            n_conv_layers=n_conv_per_block,
            activ_fn=activ_fn
        )

        if n_unet_blocks > 0:
            self.unet_block = UNetBlock(
                n_channels_in=n_channels_block,
                n_channels_out=n_channels_block,
                n_channels_block=(n_channels_block * width_factor) + width_term,
                n_conv_per_block=(n_conv_per_block * depth_factor) + depth_term,
                n_unet_blocks=n_unet_blocks - 1,
                width_factor=width_factor,
                width_term=width_term,
                depth_factor=depth_factor,
                depth_term=depth_term,
                activ_fn=activ_fn,
                skip_connect=skip_connect
            )
            self.conv_block_out = ConvBlock(
                n_channels_in=n_channels_block * (1, 2)[skip_connect],
                n_channels_block=n_channels_block,
                n_channels_out=n_channels_block,
                n_conv_layers=n_conv_per_block,
                activ_fn=activ_fn
            )
        else:
            self.unet_block = None

        self.conv_out = nn.Conv3d(
            in_channels=n_channels_block,
            out_channels=n_channels_out,
            kernel_size=(1, 1, 4),
            stride=(1, 1, 4),
            padding=(0, 0, 0)
        )

        self.linear_u = nn.Linear(n_channels_out, 3 + 1 + 1)
        self.linear_mu = nn.Linear(n_channels_out, 3 + 1 + 1)

        self.omega = omega
        self.alpha = alpha
        self.skip_connect = skip_connect
        self.regularizer = None

    def forward(self, inputs):
        '''
        Args:
            a: (batch_size, n_x, n_y, n_z, n_channels_in)
            x: (batch_size, n_x, n_y, n_z, 3)
        Returns:
            u_pred: (batch_size, n_x, n_y, n_z, 1)
            mu_pred: (batch_size, n_x, n_y, n_z, 1)
        '''
        a, x = inputs

        a = torch.permute(a, (0, 4, 1, 2, 3))
        h = self.conv_in(a)
        h = self.conv_block_in(h)

        if self.unet_block:
            if self.skip_connect:
                h = torch.cat([h, self.unet_block(h)], dim=1)
            else:
                h = self.unet_block(h)
            h = self.conv_block_out(h)

        h = self.conv_out(h)
        h = torch.permute(h, (0, 2, 3, 4, 1))
        shape = h.shape[:-1]

        u_params = self.linear_u(h)
        u_freq  = u_params[...,0:3] * self.omega
        u_phase = u_params[...,3:4] * self.omega
        u_amp   = u_params[...,4:5] # TODO not used

        u_dot = torch.einsum('bxyzd,bxyzd->bxyz', x, u_freq)[...,None]
        u_pred = torch.sin(2 * np.pi * (u_dot + u_phase)) * self.alpha

        mu_pred = u_pred * 0 # TODO

        return u_pred, mu_pred, u_dot, u_phase, u_amp


class UNetBlock(torch.nn.Module):

    def __init__(
        self,
        n_channels_in,
        n_channels_out,
        n_channels_block,
        n_conv_per_block,
        n_unet_blocks,
        activ_fn,
        skip_connect,
        width_factor=1,
        width_term=0,
        depth_factor=1,
        depth_term=0,
    ):
        super().__init__()

        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv_block_in = ConvBlock(
            n_channels_in=n_channels_in,
            n_channels_block=n_channels_block,
            n_channels_out=n_channels_block,
            n_conv_layers=n_conv_per_block,
            activ_fn=activ_fn
        )

        if n_unet_blocks > 0:
            self.unet_block = UNetBlock(
                n_channels_in=n_channels_block,
                n_channels_out=n_channels_block,
                n_channels_block=(n_channels_block * width_factor) + width_term,
                n_conv_per_block=(n_conv_per_block * depth_factor) + depth_term,
                n_unet_blocks=n_unet_blocks - 1,
                width_factor=width_factor,
                width_term=width_term,
                depth_factor=depth_factor,
                depth_term=depth_term,
                activ_fn=activ_fn,
                skip_connect=skip_connect
            )
            self.conv_block_out = ConvBlock(
                n_channels_in=n_channels_block * (1, 2)[skip_connect],
                n_channels_block=n_channels_block,
                n_channels_out=n_channels_out,
                n_conv_layers=n_conv_per_block,
                activ_fn=activ_fn
            )
        else:
            self.unet_block = None
            self.conv_block_out = ConvBlock(
                n_channels_in=n_channels_block,
                n_channels_block=n_channels_block,
                n_channels_out=n_channels_out,
                n_conv_layers=n_conv_per_block,
                activ_fn=activ_fn
            )

        self.upsample = nn.Upsample(scale_factor=2)
        self.skip_connect = skip_connect

    def forward(self, a):
        '''
        Args:
            a: (batch_size, n_channels_in, n_x, n_y, n_z)
        Returns:
            h: (batch_size, n_channels_out, n_x, n_y, n_z)
        '''
        h = self.downsample(a)
        h = self.conv_block_in(h)
        if self.unet_block:
            if self.skip_connect:
                h = torch.cat([h, self.unet_block(h)], dim=1)
            else:
                h = self.unet_block(h)
        h = self.conv_block_out(h)
        h = self.upsample(h)
        return h


class ConvBlock(torch.nn.Module):

    def __init__(
        self, n_channels_in, n_channels_block, n_channels_out, n_conv_layers, activ_fn
    ):
        super().__init__()
        self.convs = []
        for i in range(n_conv_layers):
            if i + 1 == n_conv_layers:
                n_channels_block = n_channels_out
            conv = nn.Conv3d(n_channels_in, n_channels_block, kernel_size=3, padding=1)
            self.convs.append(conv)
            self.add_module(f'conv{i}', conv)
            n_channels_in = n_channels_block
        self.activ_fn = get_activ_fn(activ_fn)

    def forward(self, h):
        for conv in self.convs:
            h = self.activ_fn(conv(h))
        return h


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

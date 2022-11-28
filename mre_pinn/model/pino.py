import time
import numpy as np
import torch
#torch.backends.cudnn.enabled = False
from torch import nn
from torch.nn import functional as F

from ..utils import as_iterable, print_if
from .generic import get_activ_fn, ParallelNet


def xavier(fan_in, fan_out, gain=1, c=6):
    scale = gain * np.sqrt(c / (fan_in + fan_out))
    w = torch.rand(fan_in, fan_out, device='cuda', dtype=torch.float32)
    return nn.Parameter((2 * w - 1) * scale)


def metadata(**kwargs):
    '''
    Print shape and type.
    '''
    for key, val in kwargs.items():
        print(f'{key}\t{val.shape}\t{val.dtype}')


def describe(**kwargs):
    '''
    Print descriptive statistics.
    '''
    for key, val in kwargs.items():
        print(f'{key}\t{val.mean()}\t{val.std()}')


class MREPINO(torch.nn.Module):

    def __init__(
        self,
        dataset,
        n_channels_block,
        n_conv_per_block,
        n_conv_blocks,
        activ_fn,
        n_latent,
        n_pinn_layers,
        n_pinn_hidden,
        conditional,
        parallel,
        omega,
        polar_input=False,
        complex_output=False,
        width_factor=1,
        width_term=0,
        depth_factor=1,
        depth_term=0,
        skip_connect=True,
        dense=True,
        debug=False
    ):
        super().__init__()

        metadata = dataset.metadata
        metadata = metadata.reset_index().groupby(['variable', 'dimension']).mean()
        self.x_loc = torch.tensor(metadata['center'].wave, dtype=torch.float32)
        self.x_scale = torch.tensor(metadata['extent'].wave, dtype=torch.float32)

        stats = dataset.describe()
        stats = stats.reset_index().groupby(['variable', 'component']).mean()
        self.u_loc = torch.tensor(stats['mean'].wave, dtype=torch.float32)
        self.u_scale = torch.tensor(stats['std'].wave, dtype=torch.float32)
        self.mu_loc = torch.tensor(stats['mean'].mre, dtype=torch.float32)
        self.mu_scale = torch.tensor(stats['std'].mre, dtype=torch.float32)
        self.a_loc = torch.tensor(stats['mean'].anat, dtype=torch.float32)
        self.a_scale = torch.tensor(stats['std'].anat, dtype=torch.float32)
        self.omega = torch.tensor(omega, dtype=torch.float32)

        if debug:
            print(self.x_loc.shape)
            print(self.u_loc.shape)
            print(self.a_loc.shape)
            print(self.mu_loc.shape)

        n_channels_block = as_iterable(n_channels_block)
        n_conv_blocks = as_iterable(n_conv_blocks)
        n_conv_per_block = as_iterable(n_conv_per_block)

        self.u_cnn = CNN(
            xyz_shape_in=dataset[0].wave.field.spatial_shape,
            n_channels_in=len(self.u_loc),
            n_channels_block=n_channels_block[0],
            n_conv_per_block=n_conv_per_block[0],
            n_conv_blocks=n_conv_blocks[0],
            activ_fn=activ_fn,
            n_output=n_latent,
            width_factor=width_factor,
            width_term=width_term,
            depth_factor=depth_factor,
            depth_term=depth_term,
            skip_connect=skip_connect,
            debug=debug
        )

        if conditional:
            self.a_cnn = CNN(
                xyz_shape_in=dataset[0].anat.field.spatial_shape,
                n_channels_in=len(self.a_loc),
                n_channels_block=n_channels_block[1],
                n_conv_per_block=n_conv_per_block[1],
                n_conv_blocks=n_conv_blocks[1],
                activ_fn=activ_fn,
                n_output=n_latent,
                width_factor=width_factor,
                width_term=width_term,
                depth_factor=depth_factor,
                depth_term=depth_term,
                skip_connect=skip_connect,
                debug=debug
            )

        self.u_pinn = HyperPINN(
            n_input=len(self.x_loc),
            n_output=len(self.u_loc),
            n_latent=n_latent * 2 if conditional and not parallel else n_latent,
            n_layers=n_pinn_layers,
            n_hidden=n_pinn_hidden,
            dense=dense,
            polar_input=polar_input,
            complex_output=complex_output,
            polar_output=False
        )
        self.mu_pinn = HyperPINN(
            n_input=len(self.x_loc),
            n_output=len(self.mu_loc),
            n_latent=n_latent * 2 if conditional and not parallel else n_latent,
            n_layers=n_pinn_layers,
            n_hidden=n_pinn_hidden,
            dense=dense,
            polar_input=polar_input,
            complex_output=complex_output,
            polar_output=True
        )
        self.regularizer = None
        self.conditional = conditional
        self.parallel = parallel

    def forward(self, inputs, debug=False, return_weights=False):
        a, u, x = inputs

        a = (a - self.a_loc) / self.a_scale
        u = (u - self.u_loc) / self.u_scale
        x = (x - self.x_loc) / self.x_scale
        x = x * self.omega

        h_u = self.u_cnn(u, debug=debug)

        if self.conditional:
            h_a = self.a_cnn(a, debug=debug)

            if self.parallel:
                h_mu = h_a
            else:
                h_u = h_mu = torch.cat([h_u, h_a], dim=-1)
        else:
            h_mu = h_u
            
        u_pred, u_weights, u_biases = self.u_pinn(h_u, x, debug=debug)
        mu_pred, mu_weights, mu_biases = self.mu_pinn(h_mu, x, debug=debug)

        u_pred  = u_pred  * self.u_scale  + self.u_loc
        mu_pred = mu_pred * self.mu_scale + self.mu_loc

        if return_weights:
            return (u_pred, mu_pred), (u_weights, mu_weights), (u_biases, mu_biases)
        else:
            return u_pred, mu_pred


class CNN(torch.nn.Module):
    '''
    Convolutional neural network.
    '''
    def __init__(
        self,
        xyz_shape_in,
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
        xyz_shape = np.array(xyz_shape_in)
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
        print_if(debug, h.shape)
        for i, (block, pool) in enumerate(zip(self.blocks, self.pools)):
            g = block(h)
            print_if(debug, g.shape)
            if self.skip_connect:
                g = torch.cat([h, g], dim=1)
            h = pool(g)
            print_if(debug, h.shape)

        h = h.reshape(h.shape[0], -1)
        print_if(debug, h.shape)
        return self.linear_out(h)


class HyperLinear(torch.nn.Module):
    '''
    Linear layer with parameters that are
    conditioned on an input latent vector.
    '''
    def __init__(self, n_latent, n_features_in, n_features_out, first=False):
        super().__init__()
        self.linear_w = nn.Linear(n_latent, n_features_out * n_features_in)
        self.linear_b = nn.Linear(n_latent, n_features_out)
        self.w_norm = nn.LayerNorm(n_features_out * n_features_in)
        self.b_norm = nn.LayerNorm(n_features_out)
        self.w_std = 1 / n_features_in if first else np.sqrt(6 / n_features_in)
        self.b_std = 1e-9

    def forward(self, h, x, debug=False):
        '''
        Args:
            h: (batch_size, n_latent)
            x: (batch_size, n_points, n_features_in)
        Returns:
            y: (batch_size, n_points, n_features_out)
        '''
        w = 2 * normal_cdf(self.w_norm(self.linear_w(h))) - 1
        b = 2 * normal_cdf(self.b_norm(self.linear_b(h))) - 1
        n_features_out = b.shape[1]
        n_features_in = w.shape[1] // n_features_out
        w = w.view(-1, n_features_in, n_features_out) * self.w_std
        b = b.view(-1, 1, n_features_out) * self.b_std
        return torch.einsum('bxi,bio->bxo', x, w) + b, w.detach(), b.detach()


def normal_cdf(x, loc=0, scale=1):
    return 0.5 * (1 + torch.erf((x - loc) / scale / np.sqrt(2)))


class HyperPINN(torch.nn.Module):

    def __init__(
        self,
        n_input,
        n_output,
        n_latent,
        n_layers,
        n_hidden,
        dense=True,
        polar_input=False,
        complex_output=False,
        polar_output=False
    ):
        assert n_layers > 0
        super().__init__()

        if polar_input:
            n_input += 3

        first = True
        self.hiddens = []
        for i in range(n_layers - 1):
            hidden = HyperLinear(n_latent, n_input, n_hidden, first)
            if dense:
                n_input += n_hidden
            else:
                n_input = n_hidden
            self.hiddens.append(hidden)
            self.add_module(f'hidden{i}', hidden)
            first = False

        if complex_output:
            self.output = HyperLinear(n_latent, n_input, n_output * 2, first)
        else:
            self.output = HyperLinear(n_latent, n_input, n_output, first)

        self.dense = dense
        self.polar_input = polar_input
        self.complex_output = complex_output
        self.polar_output = polar_output

    def forward(self, h, x, debug=False):
        '''
        Args:
            h: (batch_size, n_latent)
            x: (batch_size, n_points, n_input)
        Returns:
            y: (batch_size, n_points, n_output)
        '''
        if self.polar_input: # polar coordinates
            x, y, z = torch.split(x, 1, dim=-1)
            r = torch.sqrt(x**2 + y**2)
            sin, cos = x / (r + 1), y / (r + 1)
            x = torch.cat([x, y, z, r, sin, cos], dim=-1)

        # hidden layers
        weights = []
        biases = []
        for i, hidden in enumerate(self.hiddens):
            a, w, b = hidden(h, x, debug=debug)
            weights.append(w)
            biases.append(b)
            y = torch.sin(a)
            if self.dense:
                x = torch.cat([x, y], dim=-1)
            else:
                x = y

        # output layer
        y, w, b = self.output(h, x, debug=debug)

        if self.complex_output:
            y = as_complex(y, polar=self.polar_output)
        
        return y, weights, biases


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

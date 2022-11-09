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
        depth_term=0,
        is_unet_block=False
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
        Hidden:
            s: (batch_size, n_x, n_y, n_z, n_spatial_freqs)
            U: (batch_size, n_x, n_y, n_z, 1)
            M: (batch_size, n_x, n_y, n_z, 1)
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

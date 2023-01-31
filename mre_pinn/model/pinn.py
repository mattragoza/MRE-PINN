import numpy as np
import torch

from ..utils import as_iterable, as_complex, concat, exists
from .generic import get_activ_fn, ParallelNet


class MREPINN(torch.nn.Module):

    def __init__(self, example, omega, **kwargs):
        super().__init__()

        metadata = example.metadata
        x_center = torch.tensor(metadata['center'].wave, dtype=torch.float32)
        x_extent = torch.tensor(metadata['extent'].wave, dtype=torch.float32)

        stats = example.describe()
        self.u_loc = torch.tensor(stats['mean'].wave)
        self.u_scale = torch.tensor(stats['std'].wave)
        self.mu_loc = torch.tensor(stats['mean'].mre)
        self.mu_scale = torch.tensor(stats['std'].mre)
        self.omega = torch.tensor(omega)

        if 'anat' in example:
            self.a_loc = torch.tensor(stats['mean'].anat)
            self.a_scale = torch.tensor(stats['std'].anat)
        else:
            self.a_loc = torch.zeros(0)
            self.a_scale = torch.zeros(0)

        self.input_loc = x_center
        self.input_scale = x_extent

        self.u_pinn = PINN(
            n_input=len(self.input_loc),
            n_output=len(self.u_loc),
            complex_output=example.wave.field.is_complex,
            polar_output=False,
            **kwargs
        )
        self.mu_pinn = PINN(
            n_input=len(self.input_loc),
            n_output=len(self.mu_loc) + len(self.a_loc),
            complex_output=example.mre.field.is_complex,
            polar_output=True,
            **kwargs
        )
        self.regularizer = None

    def forward(self, inputs):
        x, = inputs
        x = (x - self.input_loc) / self.input_scale
        x = x * self.omega

        u_pred = self.u_pinn(x)
        u_pred = u_pred * self.u_scale + self.u_loc

        mu_a_pred = self.mu_pinn(x)
        mu_pred, a_pred = (
            mu_a_pred[:,:len(self.mu_loc)],
            mu_a_pred[:,len(self.mu_loc):]
        )
        mu_pred = mu_pred * self.mu_scale + self.mu_loc
        a_pred = a_pred * self.a_scale + self.a_loc

        return u_pred, mu_pred, a_pred


class PINN(torch.nn.Module):

    def __init__(
        self,
        n_input,
        n_output,
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

        self.hiddens = []
        for i in range(n_layers - 1):
            hidden = torch.nn.Linear(n_input, n_hidden)
            if dense:
                n_input += n_hidden
            else:
                n_input = n_hidden
            self.hiddens.append(hidden)
            self.add_module(f'hidden{i}', hidden)

        if complex_output:
            self.output = torch.nn.Linear(n_input, n_output * 2)
        else:
            self.output = torch.nn.Linear(n_input, n_output)

        self.dense = dense
        self.polar_input = polar_input
        self.complex_output = complex_output
        self.polar_output = polar_output

        self.init_weights()

    def forward(self, x):
        '''
        Args:
            x: (n_points, n_input)
        Returns:
            u: (n_points, n_output)
        '''
        if self.polar_input: # polar coordinates
            x, y, z = torch.split(x, 1, dim=-1)
            r = torch.sqrt(x**2 + y**2)
            sin, cos = x / (r + 1), y / (r + 1)
            x = torch.cat([x, y, z, r, sin, cos], dim=-1)

        # hidden layers
        for i, hidden in enumerate(self.hiddens):
            y = torch.sin(hidden(x))
            if self.dense:
                x = torch.cat([x, y], dim=1)
            else:
                x = y
        
        # output layer
        if self.complex_output:
            return as_complex(self.output(x), polar=self.polar_output)
        else:
            return self.output(x)

    def init_weights(self, c=6):
        '''
        SIREN weight initialization.
        '''
        for i, module in enumerate(self.children()):
            if not hasattr(module, 'weight'):
                continue
            n_input = module.weight.shape[-1]

            if i == 0: # first layer
                w_std = 1 / n_input
            else:
                w_std = np.sqrt(c / n_input)

            with torch.no_grad():
                module.weight.uniform_(-w_std, w_std)

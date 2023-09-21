import numpy as np
import torch

from ..utils import as_iterable, as_complex, concat, exists
from .generic import get_activ_fn, ParallelNet


class MREPINN(torch.nn.Module):

    def __init__(self, example, omega0, activ_fn='ss', **kwargs):
        super().__init__()

        self.omega0 = omega0
        self.is_complex = example.wave.field.is_complex
        self.x_dim = example.wave.field.n_spatial_dims
        self.u_dim = example.wave.field.n_components
        self.mu_dim = 1

        if 'anat' in example:
            self.a_dim = example.anat.field.n_components
        else:
            self.a_dim = 0

        self.u_pinn = PINN(
            n_input=self.x_dim,
            n_output=self.u_dim,
            complex_output=self.is_complex,
            polar_output=False,
            activ_fn=activ_fn[0],
            **kwargs
        )
        self.mu_pinn = PINN(
            n_input=self.x_dim,
            n_output=self.mu_dim + self.a_dim,
            complex_output=self.is_complex,
            polar_output=True,
            activ_fn=activ_fn[1],
            **kwargs
        )
        self.regularizer = None

    def forward(self, inputs):
        x, = inputs
        x = x * self.omega0
        u_pred = self.u_pinn(x)
        mu_pred, a_pred = torch.split(
            self.mu_pinn(x), (self.mu_dim, self.a_dim), dim=1
        )
        return u_pred, mu_pred, a_pred


class PINN(torch.nn.Module):

    def __init__(
        self,
        n_input,
        n_output,
        n_layers,
        n_hidden,
        activ_fn='s',
        init_sin=False,
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

        self.activ_fn = get_activ_fn(activ_fn)
        self.init_sin = init_sin
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
            if i == 0 and self.init_sin:
                y = torch.sin(hidden(x))
            else:
                y = self.activ_fn(hidden(x))
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
                if i == 0:
                    module.weight.uniform_(-w_std, w_std)
                else:
                    module.weight.uniform_(-w_std, w_std)

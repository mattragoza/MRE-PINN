import numpy as np
import torch

from .utils import identity, as_iterable, as_complex


def get_activ_fn(key):
    return {
        's': torch.sin,
        'r': torch.nn.functional.leaky_relu, 
        't': torch.tanh
    }[key]


def complex_uniform_(t, loc, scale):
    print('complex_uniform_')
    radius = torch.rand(t.shape) * scale 
    angle  = torch.rand(t.shape) * 2 * np.pi
    t[...] = radius * torch.exp(1j * angle) + loc
    return t


class PINN(torch.nn.Module):
    '''
    A physics-informed neural network for elasticity reconstruction.
    '''
    def __init__(
        self,
        n_input,
        n_outputs,
        omega0,
        n_layers,
        n_hidden,
        activ_fn,
        parallel=True,
        dense=True,
        dtype=None
    ):
        super().__init__()
        self.n_outputs = n_outputs
        self.input_scaler = InputScaler(dtype)

        if parallel:
            net_outputs = [n for n in n_outputs]
        else:
            net_outputs = [sum(n_outputs)]

        self.idxs = [0] + list(np.cumsum(n_outputs))

        # construct the network
        self.nets = [
            FFNN(
                n_input=n_input,
                n_layers=n_layers,
                n_hidden=n_hidden,
                n_output=n_output * (2, 1)[dtype.is_complex],
                activ_fn=activ_fn,
                omega0=omega0,
                dense=dense,
                dtype=dtype
            ) for n_output in net_outputs
        ]
        if parallel:
            self.net = Parallel(self.nets)
        else:
            self.net = self.nets[0]

        self.output_scaler = OutputScaler(dtype)
        self.regularizer = None

    def init_weights(self, input, *outputs):
        self.input_scaler.init_weights(input)
        self.output_scaler.init_weights(*outputs)
        for net in self.nets:
            net.init_weights(c=6)

    def forward(self, x):
        x = self.input_scaler(x)
        x = self.net(x)
        x = as_complex(x)
        x = self.output_scaler(x)
        return x


class FFNN(torch.nn.ModuleList):
    '''
    A generic feedforward neural network.

    Args:
        n_input: Number of input units.
        n_layers: Number of linear layers.
        n_hidden: Number of hidden units.
        n_output: Number of output units.
        activ_fn: Activation function(s).
        omega0: Input sine activation frequency.
        dense: If True, use dense connections.
    '''
    def __init__(
        self,
        n_input,
        n_layers,
        n_hidden,
        n_output,
        activ_fn,
        omega0=None,
        dense=False,
        dtype=torch.float32
    ):
        super().__init__()

        self.linears = []
        for i in range(n_layers):

            if i < n_layers - 1: # hidden layer
                linear = torch.nn.Linear(n_input, n_hidden, dtype=dtype)
            else: # output layer
                linear = torch.nn.Linear(n_input, n_output, dtype=dtype)
            self.linears.append(linear)
            self.add_module(f'linear{i}', linear)

            if dense:
                n_input += n_hidden
            else:
                n_input = n_hidden

        self.omega0 = omega0
        self.activ_fn = get_activ_fn(activ_fn)
        self.dense = dense

    def forward(self, input):

        # forward pass through hidden layers
        for i, linear in enumerate(self.linears):

            if i < len(self.linears) - 1: # hidden layer
                if i == 0 and self.omega0: # input layer
                    output = torch.sin(self.omega0 * linear(input))
                else:
                    output = self.activ_fn(linear(input))

                if self.dense: # dense connections
                    input = torch.cat([input, output], dim=1)
                else:
                    input = output

            else: # output layer
                output = linear(input)

        return output

    def init_weights(self, c=6):
        '''
        SIREN weight initialization.
        '''
        for i, module in enumerate(self.children()):
            n_input = module.weight.shape[-1]

            if i == 0: # first layer
                w_std = 1 / n_input
            else:
                w_std = np.sqrt(c / n_input)

            with torch.no_grad():
                if module.weight.dtype.is_complex:
                    complex_uniform_(module.weight, 0, w_std)
                else:
                    module.weight.uniform_(-w_std, w_std)


class Parallel(torch.nn.ModuleList):
    '''
    A parallel container. Applies the forward pass of each child module
    to the input and then concatenates their output along the second dim.
    '''
    def forward(self, input):
        return torch.cat([module(input) for module in self], dim=1)


class InputScaler(torch.nn.Module):

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def init_weights(self, data):
        data = torch.as_tensor(data, dtype=self.dtype)
        self.loc = data.mean(dim=0, keepdim=True)
        self.scale = data.std(dim=0, keepdim=True)

        # avoid division by zero
        self.scale[self.scale == 0] = 1

    def forward(self, input):
        return (input - self.loc) / self.scale


class OutputScaler(torch.nn.Module):

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def init_weights(self, *data):
        data = torch.cat([torch.as_tensor(d, dtype=self.dtype) for d in data], dim=1)
        self.loc = data.mean(dim=0, keepdim=True)
        self.scale = data.std(dim=0, keepdim=True)

    def forward(self, input):
        return input * self.scale + self.loc

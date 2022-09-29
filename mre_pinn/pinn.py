import numpy as np
import torch

from .utils import (
    identity, is_iterable, as_iterable, as_complex, as_real, concat, exists
)


def get_activ_fn(key):
    return {
        's': torch.sin,
        'r': torch.nn.functional.leaky_relu, 
        't': torch.tanh,
        'k': torch.nn.functional.tanhshrink,
    }[key]


class PINN(torch.nn.ModuleList):
    '''
    Physics-informed neural network.

    Args:
        n_inputs: Number of input units.
        n_layers: Number of linear layers.
        n_hidden: Number of hidden units.
        n_output: Number of output units.
        activ_fn: Activation function(s).
        omega0: Input sine activation frequency.
        dense: If True, use dense connections.
    '''
    def __init__(
        self,
        n_inputs,
        n_layers,
        n_hidden,
        n_output,
        activ_fn,
        omega0=None,
        dense=False,
        dtype=torch.float32
    ):
        super().__init__()
        self.n_input = sum(as_iterable(n_inputs))
        self.input_scaler = InputScaler(dtype)

        n_input = self.n_input
        self.linears = []
        for i in range(n_layers):

            if i < n_layers - 1: # hidden layer
                linear = torch.nn.Linear(n_input, n_hidden, dtype=dtype)
            else: # output layer
                linear = torch.nn.Linear(n_input, n_output * 2, dtype=dtype)
            self.linears.append(linear)
            self.add_module(f'linear{i}', linear)

            if dense:
                n_input += n_hidden
            else:
                n_input = n_hidden

        self.n_output = n_output
        self.output_scaler = OutputScaler(dtype)

        self.omega0 = torch.nn.Parameter(
            torch.as_tensor(omega0, dtype=dtype)
        )
        self.activ_fn = get_activ_fn(activ_fn)
        self.dense = dense

    def forward(self, inputs):

        # TODO make conditional an optional arg
        input = torch.cat(inputs, dim=-1)
        input = self.input_scaler(input)

        # forward pass through hidden layers
        for i, linear in enumerate(self.linears):

            if i < len(self.linears) - 1: # hidden layer

                if i == 0 and exists(self.omega0): # sine input layer
                    output = torch.sin(self.omega0 * linear(input))
                else:
                    output = self.activ_fn(linear(input))

                if self.dense: # dense connections
                    input = torch.cat([input, output], dim=1)
                else:
                    input = output

            else: # output layer
                output = as_complex(linear(input))

        output = self.output_scaler(output)
        return output

    def init_weights(self, inputs, output, c=6):
        '''
        SIREN weight initialization.
        '''
        input = concat(inputs, dim=-1)
        self.input_scaler.init_weights(input)
        self.output_scaler.init_weights(output)

        for i, module in enumerate(self.children()):
            if not hasattr(module, 'weight'):
                continue
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


class ParallelNet(torch.nn.Module):
    '''
    A set of parallel networks.
    '''
    net_type = NotImplemented

    def __init__(self, n_outputs, **kwargs):
        super().__init__()

        # construct parallel networks
        self.nets = []
        for i, n_output in enumerate(as_iterable(n_outputs)):
            net = self.net_type(n_output=n_output, **kwargs)
            self.nets.append(net)
            self.add_module(f'net{i}', net)

        self.regularizer = None

    def init_weights(self, inputs, outputs):
        assert len(outputs) == len(self.nets)
        for net, output in zip(self.nets, outputs):
            net.init_weights(inputs, output)

    def forward(self, inputs):
        return tuple(net(inputs) for net in self.nets)


class ParallelPINN(ParallelNet):
    net_type = PINN


class InputScaler(torch.nn.Module):

    def __init__(self, dtype=None):
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

    def __init__(self, dtype=None):
        super().__init__()
        self.dtype = dtype

    def init_weights(self, data):
        data = torch.as_tensor(data, dtype=self.dtype)
        self.loc = data.mean(dim=0, keepdim=True)
        self.scale = data.std(dim=0, keepdim=True)

    def forward(self, input):
        return input * self.scale + self.loc

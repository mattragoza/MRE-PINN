import numpy as np
import torch

from .utils import identity, as_iterable, as_complex


def gaussian(x):
    return torch.exp(-x**2)


def get_activ_fn(key):
    return {
        's': torch.sin,
        'g': gaussian
    }[key]


class MRE_PINN(torch.nn.Sequential):
    '''
    A physics-informed neural network for elasticity reconstruction.
    '''
    def __init__(
        self,
        input,
        outputs,
        omega0,
        n_layers,
        n_hidden,
        activ_fn,
        parallel=True,
        dense=True
    ):
        n_input = input.shape[1]
        n_outputs = [o.shape[1] for o in outputs]
        self.n_outputs = n_outputs

        input_scaler = InputScaler(input)
        output_scaler = OutputScaler(*outputs)
        complex_splicer = ComplexSplicer()

        if parallel:
            net_outputs = [n * 2 for n in n_outputs]
        else:
            net_outputs = [sum(n_outputs) * 2]

        self.idxs = [0] + list(np.cumsum(n_outputs))

        # construct the network
        nets = [
            Feedforward(
                n_input=n_input,
                n_layers=n_layers,
                n_hidden=n_hidden,
                n_output=n_output,
                activ_fn=activ_fn,
                dense=dense
            ) for n_output in net_outputs
        ]
        if parallel:
            net = Parallel(nets)
        else:
            net = nets[0]

        super().__init__(input_scaler, net, complex_splicer, output_scaler)

        # initialize weights
        for n in nets:
            n.init_weights(omega0)

        net.regularizer = None


class Feedforward(torch.nn.ModuleList):
    '''
    A generic feedforward neural network.

    Args:
        n_input: Number of input units.
        n_layers: Number of linear layers.
        n_hidden: Number of hidden units.
        n_output: Number of output units.
        activ_fn: Activation function(s).
        dense: If True, use dense connections.
    '''
    def __init__(
        self,
        n_input,
        n_layers,
        n_hidden,
        n_output,
        activ_fn,
        input_fn=None,
        output_fn=None,
        dense=False,
        dtype=torch.float32
    ):
        super().__init__()

        activ_fn = [get_activ_fn(a) for a in as_iterable(activ_fn, string=True)]
        n_activ_fns = len(activ_fn)

        self.linears = []
        for i in range(n_layers):

            if i < n_layers - 1: # hidden layer
                linear = []
                for j in range(n_activ_fns):
                    linear.append(torch.nn.Linear(n_input, n_hidden, dtype=dtype))
                    self.add_module(f'linear{i}_{activ_fn[j].__name__}', linear[j])

            else: # output layer
                linear = torch.nn.Linear(n_input, n_output, dtype=dtype)
                self.add_module(f'linear{i}', linear)

            self.linears.append(linear)

            if dense:
                n_input += n_hidden
            else:
                n_input = n_hidden

        self.activ_fn = activ_fn
        self.dense = dense

    def forward(self, input):

        # forward pass through hidden layers
        for i, linear in enumerate(self.linears):

            if i < len(self.linears) - 1: # hidden layer
                output = 1
                for j, linear in enumerate(linear):
                    output *= self.activ_fn[j](linear(input))

                if self.dense: # dense connections
                    input = torch.cat([input, output], dim=1)
                else:
                    input = output

            else: # output layer
                output = linear(input)

        return output

    def init_weights(self, omega0, c=6):
        '''
        SIREN weight initialization.
        '''
        for i, module in enumerate(self.children()):
            n_input = module.weight.shape[-1]

            if i == 0: # first layer
                w_std = omega0 / n_input
            else:
                w_std = np.sqrt(c / n_input)

            with torch.no_grad():
                module.weight.uniform_(-w_std, w_std)


class Parallel(torch.nn.ModuleList):
    '''
    A parallel container. Applies the forward pass of each child module
    to the input and then concatenates their output along the second dim.
    '''
    def forward(self, input):
        return torch.cat([module(input) for module in self], dim=1)


class InputScaler(torch.nn.Module):

    def __init__(self, data):
        super().__init__()
        data = torch.as_tensor(data)
        self.loc = data.mean(dim=0, keepdim=True)
        self.scale = data.std(dim=0, keepdim=True)

        # avoid division by zero
        self.scale[self.scale == 0] = 1

    def forward(self, input):
        return (input - self.loc) / self.scale


class OutputScaler(torch.nn.Module):

    def __init__(self, *data):
        super().__init__()
        data = torch.cat([torch.as_tensor(d) for d in data], dim=1)
        self.loc = data.mean(dim=0, keepdim=True)
        self.scale = data.std(dim=0, keepdim=True)

    def forward(self, input):
        return input * self.scale + self.loc


class ComplexSplicer(torch.nn.Module):

    def forward(self, input):
        return as_complex(input)

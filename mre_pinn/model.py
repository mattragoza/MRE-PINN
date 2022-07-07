import numpy as np
import torch

from .utils import identity, as_iterable


class PINN(torch.nn.ModuleList):
    '''
    A physics-informed neural network.

    Args:
        n_input: Number of input units.
        n_layers: Number of linear layers.
        n_hidden: Number of hidden units.
        n_output: Number of output units.
        activ_fn: Activation function(s).
        input_fn: Input transformation.
        output_fn: Output transformation.
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

        activ_fn = as_iterable(activ_fn)
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
        self.input_fn = input_fn or identity
        self.output_fn = output_fn or identity
        self.dense = dense


    def forward(self, input):

        # apply input transformation
        input = self.input_fn(input)

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

        # apply output transformation
        return self.output_fn(output)

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

import numpy as np
import torch
import deepxde

dtype = deepxde.config.real(torch)


def as_complex(t):
    real, imag = t[:,::2], t[:,1::2]
    return torch.complex(real, imag)


class PINN(torch.nn.ModuleList):
    
    def __init__(
        self,
        n_input,
        n_layers,
        n_hidden,
        n_output,
        activ_fn,
        complex=False,
        dense=False,
        omega0=30
    ):
        super().__init__()

        if complex:
            n_output *= 2
            self.output_fn = as_complex
        else:
            self.output_fn = lambda x: x

        self.linears = []
        for i in range(n_layers):
            if i < n_layers - 1:
                linear = torch.nn.Linear(n_input, n_hidden)
            else:
                linear = torch.nn.Linear(n_input, n_output)
            self.linears.append(linear)
            self.add_module(f'linear{i}', linear)
            if dense:
                n_input += n_hidden
            else:
                n_input = n_hidden

        self.activ_fn = activ_fn
        self.omega0 = omega0
        self.dense = dense

    def forward(self, input):
        input = (input - self.input_loc) / self.input_scale
        
        for i, linear in enumerate(self.linears):
            omega = self.omega0 if i == 0 else 1

            if i < len(self.linears) - 1:
                output = self.activ_fn(omega * linear(input))
            else:
                output = linear(input)

            if self.dense:
                input = torch.cat([input, output], dim=1)
            else:
                input = output

        output = output * self.output_scale + self.output_loc
        return self.output_fn(output)

    def init_weights(
        self, c=6, input_loc=0, input_scale=1, output_loc=0, output_scale=1
    ):
        for i, module in enumerate(self.children()):
            n_input = module.weight.shape[-1]

            if i == 0:
                w_std = 1 / n_input
            else:
                w_std = np.sqrt(c / n_input)

            with torch.no_grad():
                module.weight.uniform_(-w_std, w_std)

        self.input_scale = torch.as_tensor(
            input_scale, device=module.weight.device
        ).unsqueeze(0)
        self.input_loc = torch.as_tensor(
            input_loc, device=module.bias.device
        ).unsqueeze(0)

        self.output_scale = torch.as_tensor(
            output_scale, device=module.weight.device
        ).unsqueeze(0)
        self.output_loc = torch.as_tensor(
            output_loc, device=module.bias.device
        ).unsqueeze(0)


class Parallel(torch.nn.ModuleList):
    
    def forward(self, input):
        outputs = [module(input) for module in self]
        return torch.cat(outputs, dim=1)

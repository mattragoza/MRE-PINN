import numpy as np
import torch
import deepxde

dtype = deepxde.config.real(torch)


def as_complex(t):
    real, imag = t[:,::2], t[:,1::2]
    return torch.complex(real, imag)


class ComplexFFN(torch.nn.ModuleList):
    
    def __init__(self, n_input, n_layers, n_hidden, n_output, activ_fn, w0=30):
        
        modules = []
        for i in range(n_layers):
            is_first_layer = (i == 0)
            is_last_layer = (i == n_layers - 1)
            linear = torch.nn.Linear(
                n_input,
                n_output*2 if is_last_layer else n_hidden
            )
            n_input += n_hidden
            modules.append(linear)
    
        self.activ_fn = activ_fn
        self.w0 = w0
        super().__init__(modules)

    def forward(self, input):

        input = (input - self.input_loc) / self.input_scale
        
        for i, module in enumerate(self):
            w = self.w0 if i == 0 else 1
            if i < len(self) - 1:
                output = self.activ_fn(w*module(input))
                input = torch.cat([input, output], dim=1)
            else:
                output = module(input)

        output = output * self.output_scale + self.output_loc
        return as_complex(output)

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

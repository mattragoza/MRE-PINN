import torch
import deepxde

dtype = deepxde.config.real(torch)


def as_complex(t):
    real, imag = t[:,::2], t[:,1::2]
    return torch.complex(real, imag)


class ComplexFFN(torch.nn.ModuleList):
    
    def __init__(self, n_input, n_layers, n_hidden, n_output, activ_fn):
        
        modules = []
        for i in range(n_layers):
            is_first_layer = (i == 0)
            is_last_layer = (i == n_layers - 1)
            linear = torch.nn.Linear(
                n_input if is_first_layer else n_hidden,
                n_output*2 if is_last_layer else n_hidden
            )
            modules.append(linear)
    
        self.activ_fn = activ_fn
        super().__init__(modules)
        
    def forward(self, input):
        
        for i, module in enumerate(self):
            if i < len(self) - 1:
                input = self.activ_fn(module(input))
            else:
                output = as_complex(module(input))
        return output


class Parallel(torch.nn.ModuleList):
    
    def forward(self, input):
        outputs = [module(input) for module in self]
        return torch.cat(outputs, dim=1)

import torch

from ..utils import as_iterable


def get_activ_fn(key):
    return {
        's': torch.sin,
        'r': torch.nn.functional.leaky_relu, 
        't': torch.tanh,
        'k': torch.nn.functional.tanhshrink,
        'g': torch.nn.functional.gelu
    }[key]


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

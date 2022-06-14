import numpy as np
import torch
import deepxde


def as_tensor(a):
    dtype = torch.complex128 #deepxde.config.real(deepxde.backend.lib)
    return deepxde.backend.as_tensor(a, dtype=dtype)


def nd_coords(shape, resolution):
    resolution = np.broadcast_to(resolution, len(shape))
    dims = [
        np.arange(d) * r for d, r in zip(shape, resolution)
    ]
    coords = np.meshgrid(*dims)
    coords = np.dstack(coords).reshape(-1, len(dims))
    center = np.mean(coords, axis=0, keepdims=True)
    return coords - center


class ImagePointSet(deepxde.icbc.PointSetBC):

    def __init__(self, image, resolution, ndim=2, component=0):

        if isinstance(image, str):
            image = np.load(image)

        self.ndim = ndim
        self.component = component
        self.n_components = (image.ndim - ndim)

        dtype = deepxde.config.real(np)
        self.points = nd_coords(image.shape[:ndim], resolution).astype(dtype)
        self.values = as_tensor(image.reshape(-1, *image.shape[ndim:]))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        comp_beg = self.component
        comp_end = self.component + self.n_components
        return outputs[beg:end,comp_beg:comp_end] - self.values

import numpy as np
import torch
import deepxde


def as_tensor(a):
	dtype = torch.complex128 #deepxde.config.real(deepxde.backend.lib)
	return deepxde.backend.as_tensor(a, dtype=dtype)


def nd_coords(shape):
    dims = [np.arange(d) for d in shape]
    coords = np.meshgrid(*dims)
    return np.dstack(coords).reshape(-1, len(dims))


class ImagePointSet(deepxde.icbc.PointSetBC):

	def __init__(self, image, ndim=2, component=0):

		if isinstance(image, str):
			image = np.load(image)

		self.ndim = ndim
		self.component = component
		self.n_components = (image.ndim - ndim)

		self.points = nd_coords(image.shape[:ndim]).astype(deepxde.config.real(np))
		self.values = as_tensor(image.reshape(-1, *image.shape[ndim:]))

	def error(self, X, inputs, outputs, beg, end, aux_var=None):
		comp_beg = self.component
		comp_end = self.component + self.n_components
		return outputs[beg:end,comp_beg:comp_end] - self.values

import numpy as np
import xarray as xr
import torch
import deepxde


def nd_coords(coords, center=False):
    '''
    Cartesian product of a set of coordinate arrays.

    Args:
        coords: A list of N 1D coordinate arrays.
        center: If True, center the coordinates.
    Returns:
        An array containing N-dimensional coordinates.
    '''
    coords = np.meshgrid(*coords, indexing='ij')
    coords = np.stack(coords, axis=-1).reshape(-1, len(coords))
    if center:
        coords -= np.mean(coords, axis=0, keepdims=True)
    return coords


@xr.register_dataset_accessor('field')
@xr.register_dataarray_accessor('field')
class FieldAccessor(object):
    '''
    Accessor for treating an xarray as a scalar or vector field.
    '''
    def __init__(self, xarray):
        self.xarray = xarray

    @property
    def dims(self):
        return [d for d in self.xarray.dims if d != 'component']

    @property
    def spatial_dims(self):
        return [d for d in 'xyz' if d in self.xarray.dims]

    def points(self):
        return nd_coords((self.xarray.coords[d] for d in self.xarray.field.dims))

    def values(self):
        if 'component' in self.xarray.dims: # vector field
            n_components = self.xarray.sizes['component']
            T = self.xarray.field.dims + ['component']
            return self.xarray.transpose(*T).values.reshape(-1, n_components)
        else: # scalar field
            return self.xarray.values.reshape(-1, 1)


class VectorFieldBC(deepxde.icbc.PointSetBC):

    def __init__(
        self, points, values, component=0, batch_size=None, shuffle=True
    ):
        self.points = np.asarray(points)
        self.values = torch.as_tensor(values)
        self.component = component # which component of model output
        self.set_batch_size(batch_size, shuffle)

    def __len__(self):
        return self.points.shape[0]

    def set_batch_size(self, batch_size, shuffle=True):
        self.batch_size = batch_size
        if batch_size is not None: # batch iterator and state
            self.batch_sampler = deepxde.data.BatchSampler(len(self), shuffle=shuffle)
            self.batch_indices = None

    def collocation_points(self, X):
        if self.batch_size is not None:
            self.batch_indices = self.batch_sampler.get_next(self.batch_size)
            return self.points[self.batch_indices]
        return self.points

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        if self.batch_size is not None:
            values = self.values[self.batch_indices]
        else:
            values = self.values
        return (
            outputs[beg:end, self.component:self.component + self.values.shape[-1]] -
            values
        )

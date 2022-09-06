import numpy as np
import xarray as xr
import torch
import deepxde


def nd_coords(coords, reshape=True, standardize=False):
    '''
    Cartesian product of a set of coordinate arrays.

    Args:
        coords: A list of N 1D coordinate arrays.
        reshape: If True, reshape as a matrix.
        standardize: If True, standardize to [-1, 1]/.
    Returns:
        An array containing N-dimensional coordinates.
    '''
    coords = np.meshgrid(*coords, indexing='ij')
    coords = np.stack(coords, axis=-1)

    if reshape:
        coords = coords.reshape(-1, coords.shape[-1])

    if standardize:
        axes = tuple(range(0, coords.ndim - 1))
        loc = np.mean(coords, axis=axes, keepdims=True)
        max_ = np.max(coords, axis=axes, keepdims=True)
        min_ = np.min(coords, axis=axes, keepdims=True)
        scale = (max_ - min_) / 2
        coords = (coords - loc) / scale

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
        return [d for d in self.xarray.dims if d not in {'component', 'gradient'}]

    @property
    def n_dims(self):
        return len(self.xarray.field.dims)

    @property
    def spatial_dims(self):
        return [d for d in 'xyz' if d in self.xarray.sizes]

    @property
    def spatial_axes(self):
        return [i for i, d in enumerate(self.xarray.dims) if d in 'xyz']

    @property
    def n_spatial_dims(self):
        return len(self.xarray.field.spatial_dims)

    @property
    def spatial_shape(self):
        spatial_dims = self.xarray.field.spatial_dims
        return tuple(self.xarray.sizes[d] for d in spatial_dims)

    @property
    def value_dims(self):
        return [d for d in ['component', 'gradient'] if d in self.xarray.sizes]

    @property
    def n_value_dims(self):
        return len(self.xarray.field.value_dims)

    @property
    def value_shape(self):
        value_dims = self.xarray.field.value_dims
        return tuple(self.xarray.sizes[d] for d in value_dims)

    @property
    def has_components(self):
        return 'component' in self.xarray.sizes

    @property
    def n_components(self):
        return self.xarray.sizes['component']

    @property
    def has_gradient(self):
        return 'gradient' in self.xarray.sizes
    
    @property
    def n_gradient(self):
        return self.xarray.sizes['gradient']

    def points(self, dims=None, *args, **kwargs):
        dims = dims or self.xarray.field.dims
        return nd_coords((self.xarray.coords[d] for d in dims), *args, **kwargs)

    def spatial_points(self, *args, **kwargs):
        dims = self.xarray.field.spatial_dims
        return self.xarray.field.points(dims, *args, **kwargs)

    def values(self):
        has_components = self.xarray.field.has_components
        has_gradient = self.xarray.field.has_gradient

        if has_components and has_gradient: # tensor field
            n_components = self.xarray.field.n_components
            n_gradient = self.xarray.field.n_gradient
            T = self.xarray.field.dims + ['component', 'gradient']
            return (
                self.xarray.transpose(*T).values.reshape(-1, n_components, n_gradient)
            )

        elif has_components: # vector field
            n_components = self.xarray.field.n_components
            T = self.xarray.field.dims + ['component']
            return self.xarray.transpose(*T).values.reshape(-1, n_components)

        elif has_gradient: # covector field
            n_gradient = self.xarray.field.n_gradient
            T = self.xarray.field.dims + ['gradient']
            return self.xarray.transpose(*T).values.reshape(-1, n_gradient)

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

from itertools import product
import numpy as np
import xarray as xr
import scipy.ndimage

from .baseline import filters


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
        return [d for d in self.xarray.sizes if d not in {'component', 'gradient'}]

    @property
    def spatial_dims(self):
        return [d for d in 'xyz' if d in self.xarray.sizes]

    @property
    def spatial_axes(self):
        return [i for i, d in enumerate(self.xarray.dims) if d in set('xyz')]

    @property
    def n_spatial_dims(self):
        return len(self.xarray.field.spatial_dims)

    @property
    def non_spatial_dims(self):
        return [d for d in self.xarray.sizes if d not in set('xyz')]

    @property
    def non_spatial_axes(self):
        return [i for i, d in enumerate(self.xarray.dims) if d not in set('xyz')]

    @property
    def non_planar_axes(self):
        return [i for i, d in enumerate(self.xarray.dims) if d not in set('xy')]

    @property
    def non_xy_axes(self):
        return [i for i, d in enumerate(self.xarray.dims) if d not in set('xy')]

    @property
    def planar_dims(self):
        return [d for d in 'xy' if d in self.xarray.sizes]
    
    @property
    def origin(self):
        return [self.xarray[d].min() for d in self.spatial_dims]

    @property
    def spatial_shape(self):
        spatial_dims = self.xarray.field.spatial_dims
        return tuple(self.xarray.sizes[d] for d in spatial_dims)

    @property
    def spatial_resolution(self):
        spatial_dims = self.xarray.field.spatial_dims
        return np.array([
            self.xarray[d].diff(d).mean() for d in spatial_dims
        ])

    @property
    def planar_resolution(self):
        planar_dims = self.xarray.field.planar_dims
        return np.array([
            self.xarray[d].diff(d).mean() for d in planar_dims
        ])

    @property
    def value_dims(self):
        return [d for d in ['component', 'gradient'] if d in self.xarray.sizes]

    @property
    def n_value_dims(self):
        return len(self.xarray.field.value_dims)

    @property
    def value_shape(self):
        value_dims = self.xarray.field.value_dims
        if value_dims:
            return tuple(self.xarray.sizes[d] for d in value_dims)
        return (1,)

    @property
    def value_size(self):
        return np.prod(self.xarray.field.value_shape, dtype=int)

    @property
    def is_complex(self):
        return np.iscomplexobj(self.xarray)

    @property
    def has_frequency(self):
        return 'frequency' in self.xarray.sizes

    @property
    def n_frequencies(self):
        return self.xarray.sizes['frequency']

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
        T = self.xarray.field.spatial_dims + self.xarray.field.value_dims
        values = self.xarray.transpose(*T).values
        return values.reshape(-1, *self.xarray.field.value_shape)

    def gradient(self, dim='gradient', use_z=False, **kwargs):
        if use_z:
            spatial_dims = self.xarray.field.spatial_dims
        else:
            spatial_dims = self.xarray.field.planar_dims
        gradient = []
        for d in spatial_dims:
            g = self.xarray.field.differentiate(coord=d, use_z=use_z, **kwargs)
            gradient.append(g)
        new_dim = xr.DataArray(spatial_dims, dims=dim)
        return xr.concat(gradient, dim=new_dim)

    def divergence(self, dim='component', use_z=False, **kwargs):
        if use_z:
            spatial_dims = self.xarray.field.spatial_dims
        else:
            spatial_dims = self.xarray.field.planar_dims
        divergence = 0
        for d in self.xarray[dim].values:
            component = self.xarray.sel(**{dim: d})
            divergence += component.field.differentiate(coord=d, use_z=use_z, **kwargs)
        return divergence

    def laplacian(self, savgol=True, **kwargs):
        if savgol:
            gradient = self.xarray.field.gradient(deriv=2, savgol=True, **kwargs)
            return gradient.sum('gradient')
        else:
            gradient = self.xarray.field.gradient(deriv=1, savgol=False, **kwargs)
            return gradient.field.divergence(dim='gradient', savgol=False, **kwargs)

    def smooth(self, **kwargs):
        coord = self.xarray.field.spatial_dims[0]
        return self.xarray.field.savgol_filter(coord=coord, deriv=0, **kwargs)

    def differentiate(self, coord, savgol=True, deriv=1, **kwargs):
        if savgol:
            return self.xarray.field.savgol_filter(coord=coord, deriv=deriv, **kwargs)
        elif deriv > 1:
            d = self.xarray.differentiate(coord=coord)
            return d.field.differentiate(coord=coord, savgol=False, deriv=deriv - 1)
        else:
            return self.xarray.differentiate(coord=coord)

    def savgol_filter(self, coord, deriv, use_z=False, **kwargs):
        if use_z:
            spatial_dims = self.xarray.field.spatial_dims
            non_spatial_axes = self.xarray.field.non_spatial_axes
            spatial_resolution = self.xarray.field.spatial_resolution
        else:
            spatial_dims = self.xarray.field.planar_dims
            non_spatial_axes = self.xarray.field.non_planar_axes
            spatial_resolution = self.xarray.field.planar_resolution
        n = len(spatial_dims)
        axis = spatial_dims.index(coord)
        derivs = tuple((0, deriv)[i == axis] for i in range(n))
        kernel = filters.savgol_kernel_nd(n, **kwargs)[derivs]
        kernel = np.expand_dims(kernel, sorted(non_spatial_axes))
        #print(self.xarray.shape, kernel.shape, derivs)
        result = self.xarray.copy()
        result[...] = scipy.ndimage.convolve(self.xarray, kernel, mode='reflect')
        return result / spatial_resolution[axis]**deriv

    def fft(self, shift=True):
        '''
        Convert to spatial frequency domain.
        '''
        axes = self.xarray.field.spatial_axes
        result = xr.zeros_like(self.xarray)
        result.values = np.fft.fftn(self.xarray, norm='ortho', axes=axes)
        if shift:
            result.values = np.fft.fftshift(result.values, axes=axes)
        return result

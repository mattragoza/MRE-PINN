import sys, pathlib
import numpy as np
import xarray as xr
import scipy.io
import torch
import deepxde

from .utils import print_if, parse_iterable, as_matrix
from . import discrete


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


class PointSetBC(deepxde.icbc.PointSetBC):

    def __init__(
        self, points, values, component=0, batch_size=None, shuffle=True
    ):
        self.points = np.asarray(points)
        self.values = torch.as_tensor(values)
        self.component = component # which component of model output
        self.set_batch_size(batch_size)

    def __len__(self):
        return self.points.shape[0]

    def set_batch_size(self, batch_size):
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


def load_bioqic_dataset(
    data_root, data_name, downsample=False, frequency=None, xyz_slice=None
):
    if data_name == 'fem_box':
        data = load_bioqic_fem_box_data(data_root)
    else:
        raise ValueError(f'unrecognized data name: {data_name}')

    # select data subset
    data, ndim = select_data_subset(data, downsample, frequency, xyz_slice)
    print(data)

    # direct Helmholtz inversion via discrete laplacian
    data['Lu'] = discrete.laplacian(data['u'])
    data['Mu'] = discrete.helmholtz_inversion(data['u'], data['Lu'])

    # test on 4x downsampled data
    test_data = data.coarsen(**{d: 4 for d in data.field.spatial_dims}).mean()

    return data, test_data


def load_bioqic_fem_box_data(data_root, verbose=True):
    '''
    Args:
        data_root: Path to directory with the files:
            four_target_phantom.mat (wave image)
            fem_box_ground_truth.npy (elastogram)
    Returns:
        An xarray data set with the variables:
            u: (6, 80, 100, 10, 3) wave image.
            mu: (6, 80, 100, 10) elastogram.
        And the dimensions:
            (frequency, x, y, z, component)

        The frequencies are 50-100 Hz by 10 Hz.
        The spatial dimensions are in meters.
    '''
    data_root  = pathlib.Path(data_root)
    wave_file  = data_root / 'four_target_phantom.mat'
    elast_file = data_root / 'fem_box_ground_truth.npy'

    # load true wave image and elastogram
    u = load_mat_data(wave_file, verbose)[0]['u_ft'].T
    mu = load_np_data(elast_file, verbose)

    # spatial resolution in meters
    dx = 1e-3

    # convert to xarrays with metadata
    u_dims = ['frequency', 'component', 'z', 'x', 'y']
    u_coords = {
        'frequency': np.linspace(50, 100, u.shape[0]), # Hz
        'x': np.arange(u.shape[3]) * dx,
        'y': np.arange(u.shape[4]) * dx,
        'z': np.arange(u.shape[2]) * dx,
        'component': ['y', 'x', 'z'],
    }
    u = xr.DataArray(u, dims=u_dims, coords=u_coords) * dx

    mu_dims = ['frequency', 'z', 'x', 'y']
    mu_coords = {
        'frequency': np.linspace(50, 100, mu.shape[0]), # Hz
        'x': np.arange(mu.shape[2]) * dx,
        'y': np.arange(mu.shape[3]) * dx,
        'z': np.arange(mu.shape[1]) * dx,
    }
    mu = xr.DataArray(mu, dims=mu_dims, coords=mu_coords) # Pa

    # combine into a data set and transpose the dimensions
    data = xr.Dataset(dict(u=u, mu=mu))
    data = data.transpose('frequency', 'x', 'y', 'z', 'component')
    return data


def select_data_subset(
    data,
    downsample=None,
    frequency=None,
    xyz_slice=None
):
    '''
    Args:
        data: An xarray dataset with the dimensions:
            (frequency, x, y, z, component)
        downsample: Spatial downsampling factor.
        frequency: Single frequency to select.
        x_slice, y_slice, z_slice: Indices of spatial dimensions to subset,
            resulting in 2D or 1D.
    Returns:
        data: An xarray containing the data subset.
        ndim: Whether the subset is 1D, 2D, or 3D.
    '''
    # spatial downsampling
    if downsample and downsample > 1:
        data = data.coarsen(x=downsample, y=downsample, z=downsample).mean()

    # single frequency
    if frequency and frequency not in {'all', 'multi'}:
        print('Single frequency', end=' ')
        data = data.sel(frequency=[frequency])
    else:
        print('Multi frequency', end=' ')

    x_slice, y_slice, z_slice = parse_xyz_slice(xyz_slice)

    # single x slice
    if x_slice is not None:
        data = data.isel(x=x_slice)

    # single y slice
    if y_slice is not None:
        data = data.isel(y=y_slice)

    # single z slice
    if z_slice is not None:
        data = data.isel(z=z_slice)

    # number of spatial dimensions
    ndim = (x_slice is None) + (y_slice is None) + (z_slice is None)
    assert ndim > 0, 'no spatial dimensions'
    print(f'{ndim}D')

    # subset the displacement components
    data = data.sel(component=['z', 'y', 'x'][:ndim])

    return data, ndim


def parse_xyz_slice(xyz_slice):
    if not xyz_slice:
        return (None, None, None)
    if isinstance(xyz_slice, str):
        xyz_slice = xyz_slice.upper()
        if xyz_slice == '3D':
            return (None, None, None)
        elif xyz_slice == '2D':
            return (None, None, 0)
        elif xyz_slice == '1D':
            return (None, 75, 0)
        else:
            return map(int, xyz_slice.split('-'))
    return xyz_slice


def load_np_data(np_file, verbose=False):
    np_file = str(np_file)
    print_if(verbose, f'Loading {np_file}')
    a = np.load(np_file)
    print_if(verbose, ' '*4, type(a), a.shape, a.dtype)
    return a


def load_mat_data(mat_file, verbose=False):
    '''
    Load data set from MATLAB file.
    Args:
        mat_file: Filename, typically .mat.
        verbose: Print some info about the
            contents of the file.
    Returns:
        Loaded data in a dict-like format.
        Flag indicating MATLAB axes order.
            (i.e. if True, then reverse order)
    '''
    mat_file = str(mat_file)
    print_if(verbose, f'Loading {mat_file}')
    try:
        data = scipy.io.loadmat(mat_file)
        rev_axes = True
    except NotImplementedError as e:
        # Please use HDF reader for matlab v7.3 files
        import h5py
        data = h5py.File(mat_file)
        rev_axes = False
    except:
        print(f'Failed to load {mat_file}', file=sys.stderr)
        raise
    if verbose:
        print_mat_info(data, level=1)
    return data, rev_axes


def print_mat_info(data, level=0, tab=' '*4):
    '''
    Recursively print information
    about the contents of a data set
    stored in a dict-like format.
    '''
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(tab*level + f'{k}: {type(v)} {v.shape} {v.dtype}')
        else:
            print(tab*level + f'{k}: {type(v)}')
        if hasattr(v, 'items'):
            print_mat_info(v, level+1)


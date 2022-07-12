import pathlib
import numpy as np
import xarray as xr
import scipy.io
import torch
import deepxde


def as_tensor(a, complex=True):
    if complex:
        dtype = torch.complex64
    else:
        dtype = deepxde.config.real(deepxde.backend.lib)
    return deepxde.backend.as_tensor(a, dtype=dtype)


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


class XArrayBC(deepxde.icbc.PointSetBC):

    def __init__(
        self, xarray, component=0, batch_size=None, shuffle=True
    ):
        array = xarray.to_numpy()
        coords = [xarray.coords[d] for d in xarray.dims]
        points = nd_coords(coords[:-1])

        self.shape = array.shape
        self.points = points.astype(deepxde.config.real(np))
        self.values = as_tensor(array).reshape(-1, array.shape[-1])
        self.component = component # which output component

        # batch iterator and state
        self.batch_sampler = deepxde.data.BatchSampler(len(self), shuffle)
        self.batch_size = batch_size
        self.batch_indices = None

    def __len__(self):
        return self.points.shape[0]

    def collocation_points(self, X):
        self.batch_indices = self.batch_sampler.get_next(self.batch_size)
        return self.points[self.batch_indices]

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        return (
            outputs[beg:end, self.component:self.component + self.values.shape[-1]] -
            self.values[self.batch_indices]
        )


def load_bioqic_fem_box_data(data_root):
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
    u = load_mat_data(wave_file)[0]['u_ft'].T
    mu = np.load(elast_file)

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
    x_slice=None,
    y_slice=None,
    z_slice=None,
):
    '''
    Args:
        data: An xarray dataset with the dimensions:
            (frequency, x, y, z, component)
        downsample: Spatial downsampling factor.
        frequency: Single frequency to select.
        x_slice, y_slice, z_slice: Indices of spatial
            dimensions to subset, resulting in 2D or 1D.
    Returns:
        data: An xarray containing the data subset.
        ndim: Whether the subset is 1D, 2D, or 3D.
    '''
    assert list(data.dims) == ['frequency', 'x', 'y', 'z', 'component']

    # spatial downsampling
    if downsample and downsample > 1:
        data = data.coarsen(x=downsample, y=downsample, z=downsample).mean()

    # single frequency
    if frequency is not None:
        print('Single frequency')
        data = data.sel(frequency=[frequency])
    else:
        print('Multi frequency')

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
        print(f'Loading {mat_file}')
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


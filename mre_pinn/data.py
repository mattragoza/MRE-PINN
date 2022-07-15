import sys, pathlib
import numpy as np
import xarray as xr
import scipy.io

from .utils import print_if
from . import discrete


def load_bioqic_dataset(
    data_root, data_name, frequency=None, xyz_slice=None, downsample=2
):
    if data_name == 'fem_box':
        data = load_bioqic_fem_box_data(data_root)
    else:
        raise ValueError(f'unrecognized data name: {data_name}')

    # select data subset
    data, ndim = select_data_subset(data, frequency, xyz_slice)
    print(data)

    # direct Helmholtz inversion via discrete laplacian
    data['Lu'] = discrete.laplacian(data['u'])
    data['Mu'] = discrete.helmholtz_inversion(data['u'], data['Lu'])

    # test on 4x downsampled data
    if downsample:
        downsample = {d: downsample for d in data.field.spatial_dims}
        test_data = data.coarsen(**downsample).mean()
    else:
        test_data = data.copy()

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
    frequency=None,
    xyz_slice=None,
    downsample=None
):
    '''
    Args:
        data: An xarray dataset with the dimensions:
            (frequency, x, y, z, component)
        frequency: Single frequency to select.
        x_slice, y_slice, z_slice: Indices of spatial dimensions to subset,
            resulting in 2D or 1D.
        downsample: Spatial downsampling factor.
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


import sys, pathlib
import numpy as np
import xarray as xr
import scipy.io
import scipy.ndimage

from .utils import print_if, as_xarray
from . import discrete


def complex_normal(shape, loc, scale):
    radius = np.random.randn(*shape) * scale 
    angle = np.random.rand(*shape) * 2 * np.pi
    return radius * np.exp(1j * angle) + loc


def add_complex_noise(array, noise_ratio, axis=0):
    array_abs = np.abs(array)
    array_mean = np.mean(array_abs, axis=axis)
    array_variance = np.var(array_abs, axis=axis)
    array_power = array_mean**2 + array_variance
    noise_power = noise_ratio * array_power
    noise_std = np.sqrt(noise_power).values
    noise = complex_normal(array.shape, loc=0, scale=noise_std)
    return array + noise


def load_bioqic_dataset(
    data_root,
    data_name,
    frequency=None,
    xyz_slice=None,
    downsample=2,
    noise_ratio=0,
    baseline=True,
    verbose=True
):
    if data_name == 'fem_box':
        data = load_bioqic_fem_box_data(data_root, verbose)
    elif data_name.startswith('phantom'):
        which = data_name[len('phantom'):].strip('_')
        data = load_bioqic_phantom_data(data_root, which, verbose)
    else:
        raise ValueError(f'unrecognized data name: {data_name}')

    # select data subset
    data, ndim = select_data_subset(data, frequency, xyz_slice, verbose=verbose)

    if verbose:
        print(data)

    # convert region to a coordinate label
    #data = data.assign_coords(spatial_region=data.spatial_region)

    # add complex-valued noise to wave image
    if noise_ratio > 0:
        data['u'] = add_complex_noise(data['u'], noise_ratio)

    if baseline: # direct Helmholtz inversion via discrete laplacian
        data['Lu'] = discrete.laplacian(data['u'])
        data['Mu'] = discrete.helmholtz_inversion(data['u'], data['Lu'])

    # test on 4x downsampled data
    if downsample:
        downsample = {d: downsample for d in data.field.spatial_dims}
        test_data = data.coarsen(**downsample).mean()
        test_data['spatial_region'] = data.spatial_region.coarsen(**downsample).max()
        test_data = test_data.assign_coords(spatial_region=test_data.spatial_region)
    else:
        test_data = data.copy()

    return data, test_data


def apply_lowpass_filter(u, resolution, threshold=100, order=1):
    nax = np.newaxis
    n_x, n_y, n_z = u.shape
    k_x = -(np.arange(n_x) - np.fix(n_x / 2)) / (n_x * resolution)
    k_y = -(np.arange(n_y) - np.fix(n_y / 2)) / (n_y * resolution)
    k_z = -(np.arange(n_z) - np.fix(n_z / 2)) / (n_z * resolution)
    abs_k = np.sqrt(
        np.abs(k_x)[:,nax,nax]**2 +
        np.abs(k_y)[nax,:,nax]**2 +
        np.abs(k_z)[nax,nax,:]**2
    )
    k = 1 / (1 + (abs_k / threshold)**(2 * order))
    return np.fft.ifftn(np.fft.fft(u) * np.fft.ifftshift(k))


def load_bioqic_phantom_data(data_root, which='unwrapped_dejittered', verbose=True):
    '''
    Args:
        data_root: Path to directory with the files:
            phantom_unwrapped_dejittered.mat (wave image)
            phantom_elastogram.npy (elastogram)
            phantom_regions.npy (segmentation mask)
        which: One of the following values:
            "raw_complex", "raw", "unwrapped", or "unwrapped_dejittered" (default)
    Returns:
        An xarray data set with the variables:
            u: (6, 80, 100, 10, 3) wave image.
            mu: (6, 80, 100, 10) elastogram.
        And the dimensions:
            (frequency, x, y, z, component)

        The frequencies are 50-100 Hz by 10 Hz.
        The spatial dimensions are in meters.
    '''
    which = which or 'unwrapped_dejittered'

    data_root = pathlib.Path(data_root)
    wave_file = data_root / f'phantom_{which}.mat'
    elast_file = data_root / 'phantom_elastogram.npy'
    region_file = data_root / 'phantom_regions.npy'

    wave_var = {
        'raw_complex': 'cube',
        'raw': 'phase',
        'unwrapped': 'phase_unwrapped',
        'unwrapped_dejittered': 'phase_unwrap_noipd'
    }[which]

    # load true wave image and elastogram
    data, rev_axes = load_mat_data(wave_file, verbose)
    u = data[wave_var].T if rev_axes else data[wave_var]
    #mu = load_np_data(elast_file, verbose)
    #sr = load_np_data(region_file, verbose)

    # preprocess the wave field
    #   we estimate and remove the phase shift between images
    #   and we extract the fundamental frequency across time
    #   and we do some noise reduction
    u_median = np.median(u, axis=range(2, 6), keepdims=True)
    phase_shift = np.round(u_median / (2 * np.pi)) * (2 * np.pi)
    u = u - phase_shift

    # Gaussian filter
    u = np.exp(1j * u)
    for i in range(u.shape[0]): # driver frequency
        for j in range(u.shape[1]): # displacement component
            for k in range(u.shape[2]): # time step
                u[i,j,k] = scipy.ndimage.gaussian_filter(
                    u[i,j,k], sigma=0.65 * 2, truncate=3
                )
                u[i,j,k] /= np.abs(u[i,j,k])

    # fourier transform to extract fundamental time frequency
    u = np.fft.fftn(u, axes=[2])[:,:,1]

    # Butterworth low-pass filter
    for i in range(u.shape[0]): # driver frequency
        for j in range(u.shape[1]): # displacement component
            u[i,j] = apply_lowpass_filter(u[i,j], resolution=1.5e-3, threshold=100)

    # spatial resolution in meters
    dx = 1.5e-3

    # convert to xarrays with metadata
    u_dims = ['frequency', 'component', 'z', 'x', 'y']
    u_coords = {
        'frequency': np.linspace(30, 100, u.shape[0]), # Hz
        'component': ['z', 'x', 'y'],
        'z': np.arange(u.shape[2]) * dx,
        'x': np.arange(u.shape[3]) * dx,
        'y': np.arange(u.shape[4]) * dx
    }
    u = xr.DataArray(u, dims=u_dims, coords=u_coords)

    # combine into a data set and transpose the dimensions
    data = xr.Dataset(dict(u=u))
    data = data.transpose('frequency', 'x', 'y', 'z', 'component')

    return data


def load_bioqic_fem_box_data(data_root, verbose=True):
    '''
    Args:
        data_root: Path to directory with the files:
            four_target_phantom.mat (wave image)
            fem_box_elastogram.npy (elastogram)
            fem_box_regions.npy (segmentation mask)
    Returns:
        An xarray data set with the variables:
            u: (6, 80, 100, 10, 3) wave image.
            mu: (6, 80, 100, 10) elastogram.
        And the dimensions:
            (frequency, x, y, z, component)

        The frequencies are 50-100 Hz by 10 Hz.
        The spatial dimensions are in meters.
    '''
    data_root = pathlib.Path(data_root)
    wave_file = data_root / 'four_target_phantom.mat'
    elast_file = data_root / 'fem_box_elastogram.npy'
    region_file = data_root / 'fem_box_regions.npy'

    # load true wave image and elastogram
    data, rev_axes = load_mat_data(wave_file, verbose)
    u = data['u_ft'].T if rev_axes else data['u_ft']
    mu = load_np_data(elast_file, verbose)
    sr = load_np_data(region_file, verbose)

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

    sr_dims = ['z', 'x', 'y']
    sr_coords = {
        'z': np.arange(sr.shape[0]) * dx,
        'x': np.arange(sr.shape[1]) * dx,
        'y': np.arange(sr.shape[2]) * dx,
    }
    sr = xr.DataArray(sr, dims=sr_dims, coords=sr_coords)

    # combine into a data set and transpose the dimensions
    data = xr.Dataset(dict(u=u, mu=mu, spatial_region=sr))
    data = data.transpose('frequency', 'x', 'y', 'z', 'component')

    return data


def select_data_subset(
    data,
    frequency=None,
    xyz_slice=None,
    downsample=None,
    verbose=True
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
        downsample = {'x': downsample, 'y': downsample, 'z': downsample}
        data = data.coarsen(**downsample).mean()
        data['spatial_region'] = data.spatial_region.coarsen(**downsample).max()

    # single frequency
    if frequency and frequency not in {'all', 'multi'}:
        if verbose:
            print('Single frequency', end=' ')
        data = data.sel(frequency=[frequency])
    else:
        if verbose:
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
    if verbose:
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


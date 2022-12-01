import sys, pathlib, urllib
import numpy as np
import pandas as pd
import xarray as xr
import scipy.io
import scipy.ndimage
import skimage.draw

from .dataset import MREDataset
from ..utils import print_if, as_xarray
from ..visual import XArrayViewer


class BIOQICSample(object):
    '''
    An MRE sample dataset from https://bioqic-apps.charite.de/.
    '''
    @property
    def mat_name(self):
        raise NotImplementedError
    
    @property
    def mat_base(self):
        return self.mat_name + '.mat'

    @property
    def mat_file(self):
        return self.download_dir / self.mat_base

    def download(self, verbose=True):
        url = f'https://bioqic-apps.charite.de/DownloadSamples?file={self.mat_base}'
        print_if(verbose, f'Downloading {url}')
        self.download_dir.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, self.mat_file)

    def load_mat(self, verbose=True):
        data, rev_axes = load_mat_file(self.mat_file, verbose)

        wave = data[self.wave_var].T if rev_axes else mat[wave_var]
        wave = self.add_metadata(wave)
        self.arrays = xr.Dataset(dict(wave=wave))

        if self.anat_var is not None:
            anat = data[self.anat_var].T if rev_axes else mat[anat_var]
            anat = self.add_metadata(anat)
            self.arrays['anat'] = anat

        print_if(verbose, self.arrays)

    def preprocess(self, verbose=True):
        self.segment_regions(verbose)
        self.create_elastogram(verbose)
        self.preprocess_wave_image(verbose)

    def select_data_subset(self, frequency, xyz_slice, verbose=True):
        self.arrays, ndim = select_data_subset(
            self.arrays, frequency, xyz_slice, verbose=verbose
        )

    def spatial_downsample(self, factor, verbose=True):
        print_if(verbose, 'Spatial downsampling')
        factors = {d: factor for d in self.arrays.field.spatial_dims}
        arrays = self.arrays.coarsen(boundary='trim', **factors).mean()
        arrays['spatial_region'] = self.arrays.spatial_region.coarsen(
            boundary='trim', **factors
        ).max()
        self.arrays = arrays

    def view(self, *args, **kwargs):
        if not args:
            args = self.arrays.keys()
        for arg in args:
            viewer = XArrayViewer(self.arrays[arg], **kwargs)

    def to_dataset(self):
        return MREDataset.from_bioqic(self)


class BIOQICFEMBox(BIOQICSample):

    def __init__(self, download_dir):
        self.download_dir = pathlib.Path(download_dir)

    @property
    def mat_name(self):
        return 'four_target_phantom'

    @property
    def anat_var(self):
        return None

    @property
    def wave_var(self):
        return 'u_ft'

    def add_metadata(self, array):
        resolution = 1e-3 # meters
        dims = ['frequency', 'component', 'z', 'x', 'y']
        coords = {
            'frequency': np.arange(50, 101, 10), # Hz
            'x': np.arange(80)  * resolution,
            'y': np.arange(100) * resolution,
            'z': np.arange(10)  * resolution,
            'component': ['y', 'x', 'z'],
        }
        array = xr.DataArray(array, dims=dims, coords=coords)
        return array.transpose('frequency', 'x', 'y', 'z', 'component')

    def segment_regions(self, verbose=True):
        
        print_if(verbose, 'Segmenting spatial regions')
        u = self.arrays.wave.mean(['frequency', 'component'])

        matrix_mask = np.ones((80, 100), dtype=int)
        disk_mask = np.zeros((80, 100), dtype=int)
        disks = [
            skimage.draw.disk(center=(39.8, 73.6), radius=10),
            skimage.draw.disk(center=(39.8, 49.6), radius=5),
            skimage.draw.disk(center=(39.8, 31.8), radius=3),
            skimage.draw.disk(center=(39.8, 18.6), radius=2),
        ]
        disk_mask[disks[0]] = 1
        disk_mask[disks[1]] = 2
        disk_mask[disks[2]] = 3
        disk_mask[disks[3]] = 4

        mask = (matrix_mask + disk_mask)[:,:,np.newaxis]
        mask = as_xarray(np.broadcast_to(mask, u.shape), like=u)
        mask.name = 'spatial_region'
        self.arrays['spatial_region'] = mask
        self.arrays = self.arrays.assign_coords(
            spatial_region=self.arrays.spatial_region
        )

    def create_elastogram(self, verbose=True):

        print_if(verbose, 'Creating ground truth elastogram')
        spatial_region = self.arrays.spatial_region
        wave = self.arrays.wave

        # ground truth physical parameters
        mu = np.array(
            [0, 3e3, 10e3, 10e3, 10e3, 10e3]
        )[spatial_region][np.newaxis,...,]

        axes = tuple(range(1, mu.ndim))
        omega = 2 * np.pi * wave.frequency
        omega = np.expand_dims(omega, axis=axes)

        # Voigt model
        eta = 1 # Pa·s
        mu = mu + 1j * omega * eta
        mu = as_xarray(mu, like=wave.mean(['component']))
        mu.name = 'elastogram'
        self.arrays['mu'] = mu

    def preprocess_wave_image(self, verbose=True):
        pass


class BIOQICPhantom(BIOQICSample):

    def __init__(self, download_dir, which='unwrapped_dejittered'):
        self.download_dir = pathlib.Path(download_dir)
        self.which = which

    @property
    def mat_name(self):
        return f'phantom_{self.which}'

    @property
    def anat_var(self):
        return 'magnitude'

    @property
    def wave_var(self):
        return {
            'raw_complex': 'cube',
            'raw': 'phase',
            'unwrapped': 'phase_unwrapped',
            'unwrapped_dejittered': 'phase_unwrap_noipd'
        }[self.which]

    def add_metadata(self, array):
        resolution = 1.5e-3 # meters
        dims = ['frequency', 'component', 't', 'z', 'x', 'y']
        coords = {
            'frequency': np.arange(30, 101, 10), # Hz
            't': np.arange(8),
            'x': np.arange(128) * resolution,
            'y': np.arange(80)  * resolution,
            'z': np.arange(25)  * resolution,
            'component': ['z', 'x', 'y'],
        }
        array = xr.DataArray(array, dims=dims, coords=coords)
        return array.transpose('frequency', 't', 'x', 'y', 'z', 'component')

    def segment_regions(self, sigma=0.8, threshold=280, verbose=True):

        print_if(verbose, 'Segmenting spatial regions')
        a = self.arrays.anat.mean(['frequency', 't', 'component'])

        # distinguish phantom from background
        matrix_mask = (
            scipy.ndimage.gaussian_filter(a, sigma=sigma) > threshold
        ).astype(int)
        r, c = skimage.draw.rectangle(start=(25,15), end=(103,65))
        matrix_mask[r,c,:] = 1

        # identify cylindrical inclusions
        disk_mask = np.zeros((128, 80), dtype=int)
        disks = [
            skimage.draw.disk(center=(50.0, 31.0), radius=5),
            skimage.draw.disk(center=(77.0, 31.0), radius=5),
            skimage.draw.disk(center=(50.0, 50.5), radius=5),
            skimage.draw.disk(center=(75.0, 54.5), radius=5),
        ]
        disk_mask[disks[0]] = 1
        disk_mask[disks[1]] = 2
        disk_mask[disks[2]] = 3
        disk_mask[disks[3]] = 4

        mask = matrix_mask + disk_mask[:,:,np.newaxis]
        mask = as_xarray(mask, like=a)
        mask.name = 'spatial_region'
        self.arrays['spatial_region'] = mask
        self.arrays = self.arrays.assign_coords(
            spatial_region=self.arrays.spatial_region
        )

    def create_elastogram(self, verbose=True):

        print_if(verbose, 'Creating ground truth elastogram')
        spatial_region = self.arrays.spatial_region
        wave = self.arrays.wave
        
        # ground truth physical parameters
        #   mu is shear elasticity
        #   eta is shear viscosity
        mu = np.array(
            [0, 10830, 43301, 5228, 6001, 16281]
        )[spatial_region][np.newaxis,...] # Pa
        eta = 1 # Pa·s

        # springpot model
        #   alpha interpolates b/tw spring and dashpot
        alpha = np.array(
            [0, 0.0226, 0.0460, 0.0272, 0.0247, 0.0345]
        )[spatial_region][np.newaxis,...]

        axes = tuple(range(1, mu.ndim))
        omega = 2 * np.pi * wave.frequency
        omega = np.expand_dims(omega, axis=axes)

        # springpot shear modulus
        mu = mu**(1 - alpha) * (1j * omega * eta)**alpha

        dims = ['frequency', 'x', 'y', 'z']
        coords = {
            'frequency': wave.frequency,
            'x': wave.x,
            'y': wave.y,
            'z': wave.z
        } 
        mu = xr.DataArray(mu, dims=dims, coords=coords)
        mu.name = 'elastogram'
        self.arrays['mu'] = mu

    def preprocess_wave_image(
        self, sigma=0.65, truncate=3, threshold=100, order=1, verbose=True
    ):
        '''
        Args:
            sigma: Standard deviation for Gaussian filter.
            truncate: Truncate argument for Gaussian filter.
            threshold: Cutoff frequency for low-pass filter.
            order: Frequency roll-off rate for low-pass filter.
        '''
        print_if(verbose, 'Preprocessing wave image')

        u = self.arrays.wave
        resolution = u.field.spatial_resolution

        # we estimate and remove the phase shift between images
        #   and we extract the fundamental frequency across time
        #   and we do some noise reduction

        #u_median = u.median(dim=['t', 'x', 'y', 'z'])
        #phase_shift = (u_median / (2 * np.pi)).round() * (2 * np.pi)
        #u -= phase_shift

        # (frequency, t, x, y, z, component)
        u = u.values.astype(np.complex128)

        harmonic = 1
        for f in range(u.shape[0]): # frequency
            for c in range(u.shape[5]): # component
                for t in range(u.shape[1]): # time

                    # Gaussian phase smoothing
                    u[f,t,...,c] = smooth_phase(u[f,t,...,c], sigma, truncate)

                    # gradient-based phase unwrapping
                    u[f,t,...,c] = unwrap_phase(u[f,t,...,c], resolution)

                # Fourier transform across time
                u[f,...,c] = np.fft.fft(u[f,...,c], axis=0)

                h = harmonic
                for z in range(u.shape[4]): # z slice

                    # Butterworth low-pass filtering
                    u[f,h,...,z,c] = lowpass_filter_2d(u[f,h,...,z,c], resolution[:2])

                # broadcast selected harmonic
                u[f,:,...,c] = u[f,h:h+1,...,c]

        u = u * 1e-5 # scale hack
        self.arrays['wave'] = (self.arrays.wave.dims, u)

        # average all variables across time
        self.arrays = self.arrays.mean('t')


def smooth_phase(u, sigma=0.65, truncate=3):
    '''
    Args:
        u: (n_x, n_y, n_z) phase image.
    Returns:
        (n_x, n_y, n_z) smoothed phase image.
    '''
    u_comp = np.exp(1j * u)
    u_comp = scipy.ndimage.gaussian_filter(u_comp, sigma=sigma, truncate=truncate)
    return np.angle(u_comp)


def unwrap_phase(u, resolution, component=1):
    '''
    Args:
        u: (n_x, n_y, n_z) phase image.
    Returns:
        (n_x, n_y, n_z) unwrapped phase image.
    '''
    u_comp = np.exp( 1j * u)
    u_conj = np.exp(-1j * u)
    u_grad = np.gradient(u_comp, *resolution)[component]
    return (u_grad * u_conj).imag


def lowpass_filter_2d(u, resolution, threshold=100, order=1):
    '''
    Args:
        u: (n_x, n_y) wave field
    Returns:
        (n_x, n_y) shear wave field
    '''
    nax = np.newaxis
    n_x, n_y = u.shape
    dx, dy = resolution

    # construct the filter kernel in Fourier domain
    k_x = -(np.arange(n_x) - np.fix(n_x / 2)) / (n_x * dx)
    k_y = -(np.arange(n_y) - np.fix(n_y / 2)) / (n_y * dy)
    abs_k = np.sqrt(
        np.abs(k_x)[:,nax]**2 + np.abs(k_y)[nax,:]**2
    )
    k = 1 / (1 + (abs_k / threshold)**(2 * order))
    k = np.fft.ifftshift(k)

    # apply the filter
    u_ft = np.fft.fftn(u)
    u_ft *= k
    return np.fft.ifftn(u_ft)


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
        print_if(verbose, 'Single frequency', end=' ')
        data = data.sel(frequency=[frequency])
    else:
        print_if(verbose, 'Multi frequency', end=' ')

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
    print_if(verbose, f'{ndim}D')

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


def load_mat_file(mat_file, verbose=False):
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

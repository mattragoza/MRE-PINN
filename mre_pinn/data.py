import numpy as np
import scipy.io
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


import numpy as np
import scipy.io
import torch
import deepxde


def as_tensor(a, complex=True):
    if complex:
        dtype = torch.complex128
    else:
        dtype = deepxde.config.real(deepxde.backend.lib)
    return deepxde.backend.as_tensor(a, dtype=dtype)


def nd_coords(coords, center=False):
    coords = np.meshgrid(*coords, indexing='ij')
    coords = np.stack(coords, axis=-1).reshape(-1, len(coords))
    if center is True:
        coords -= np.mean(coords, axis=0, keepdims=True)
    return coords


def nd_coords_from_shape(shape, resolution=1, center=False):
    resolution = np.broadcast_to(resolution, len(shape))
    coords = [
        np.arange(d) * r for d, r in zip(shape, resolution)
    ]
    return nd_coords(coords, center)


class NDArrayBC(deepxde.icbc.PointSetBC):

    def __init__(
        self, array, resolution=1, component=0, batch_size=None, shuffle=True
    ):
        try: # assume it's an xarray with metadata
            coords = [array.coords[d] for d in array.dims]
            points = nd_coords(coords[:-1])
            array = np.array(array)

        except AttributeError:
            points = nd_coords_from_shape(
                array.shape[:-1], resolution
            )

        self.points = points.astype(deepxde.config.real(np))
        self.values = as_tensor(array).reshape(-1, array.shape[-1])
        self.component = component

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


from functools import wraps
import numpy as np
import xarray as xr
import torch


def identity(x):
    return x


def print_if(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def as_iterable(obj, length=1, string=False):
    iterable = (list, tuple, str) if string else (list, tuple)
    if not isinstance(obj, iterable):
        return [obj] * length
    return obj


def as_matrix(a):
    '''
    Reshape an array or tensor as a matrix.
    '''
    if a.ndim > 2:
        return a.reshape(-1, a.shape[-1])
    elif a.ndim == 2:
        return a
    elif a.ndim == 1:
        return a.reshape(-1, 1)
    else:
        return a.reshape(1, 1)


def as_complex(a, interleave=True):
    '''
    Combine the even and odd indices of a real
    array or tensor into the real and imaginary
    parts of a complex array or tensor.

    Args:
        a: (..., 2M) real-valued array/tensor.
    Returns:
        An (..., M) complex-valued array/tensor.
    '''
    if interleave:
        assert a.shape[-1] % 2 == 0
        return a[...,::2] + 1j * a[...,1::2]
    else:
        assert a.shape[-1] == 2
        return a[...,0] + 1j * a[...,1]


def as_real(a, interleave=True):
    '''
    Interleave the real and imaginary parts of a
    complex array or tensor into the even and odd
    indices of a real array or tensor.

    Args:
        a: (N, M) complex-valued array/tensor.
    Returns:
        An (N, 2M) real-valued array/tensor.
    '''
    if isinstance(a, torch.Tensor):
        a = torch.stack([a.real, a.imag], dim=-1)
    else:
        a = np.stack([a.real, a.imag], axis=-1)

    if interleave and a.ndim > 1:
        return a.reshape(*a.shape[:-2], -1)

    return a


def as_xarray(a, like):
    '''
    Convert an array to an xarray, copying the dims and coords
    of a reference xarray.

    Args:
        a: An array to convert to xarray format.
        like: The reference xarray.
    Returns:
        An xarray with the given array values.
    '''
    return xr.DataArray(a, dims=like.dims, coords=like.coords)


def copy_metadata(func):
    @wraps(func)
    def wrapper(a, *args, **kwargs):
        ret = func(a, *args, **kwargs)
        if isinstance(a, xr.DataArray):
            return as_xarray(ret, like=a)
        return ret
    return wrapper


def minibatch(func, batch_size):
    @wraps(func)
    def wrapper(*args, **kwargs):
        N = args[0].shape[0]
        if batch_size >= N:
            return func(*args, **kwargs)
        outputs = []
        for i in range(0, N, batch_size):
            batch_args = [a[i:i + batch_size] for a in args]
            batch_output = func(*batch_args, **kwargs)
            outputs.append(batch_output)
        return np.concatenate(outputs, axis=0)
    return wrapper

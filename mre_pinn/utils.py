from functools import wraps
import numpy as np
import xarray as xr
import torch


def identity(x):
    return x


def exists(x):
    return x is not None


def print_if(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def is_iterable(obj, string_ok=False):
    if isinstance(obj, str):
        return string_ok
    return hasattr(obj, '__iter__')


def as_iterable(obj, length=1, string_ok=False):
    if not is_iterable(obj, string_ok):
        return [obj] * length
    return obj


def parse_iterable(obj, sep='-', type=None):
    if isinstance(obj, str):
        obj = obj.split(sep)
    if type is not None:
        return [type(x) for x in obj]
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
    if a.dtype.is_complex:
        return a
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


def as_xarray(a, like, suffix=None):
    '''
    Convert an array to an xarray, copying the dims and coords
    of a reference xarray.

    Args:
        a: An array to convert to xarray format.
        like: The reference xarray.
    Returns:
        An xarray with the given array values.
    '''
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if suffix is not None:
        name = like.name + suffix
    else:
        name = like.name
    return xr.DataArray(a, dims=like.dims, coords=like.coords, name=name)


def copy_metadata(func, suffix=None):
    @wraps(func)
    def wrapper(a, *args, **kwargs):
        ret = func(a, *args, **kwargs)
        if isinstance(a, xr.DataArray):
            return as_xarray(ret, like=a, suffix=suffix)
        return ret
    return wrapper


def concat(args, dim=0):
    try:
        return torch.cat(args, dim=dim)
    except TypeError:
        return np.concatenate(args, axis=dim)


def minibatch(method):

    @wraps(method)
    def wrapper(self, *args, batch_size=None, **kwargs):

        N = args[0].shape[0]
        assert N > 0

        if batch_size is None or batch_size >= N:
            return method(self, *args, **kwargs)

        outputs = []
        for i in range(0, N, batch_size):
            batch_args = [a[i:i + batch_size] for a in args]
            batch_output = method(self, *batch_args, **kwargs)
            outputs.append(batch_output)

        if isinstance(batch_output, tuple):
            return map(concat, zip(*outputs))

        return concat(outputs)

    return wrapper


def main(func):
    import sys, inspect, argparse

    parent_frame = inspect.stack()[1].frame
    __name__ = parent_frame.f_locals.get('__name__')

    if __name__ == '__main__':

        # get full argument specification
        argspec = inspect.getfullargspec(func)
        args = argspec.args or []
        defaults = argspec.defaults or ()
        undefined = object() # sentinel object
        n_undefined = len(args) - len(defaults)
        defaults = (undefined,) * n_undefined + defaults

        # automatically generate argument parser
        parser = argparse.ArgumentParser()
        for name, default in zip(argspec.args, defaults):
            type_ = argspec.annotations.get(name, None)
            
            if default is undefined: # positional argument
                parser.add_argument(name, type=type_)

            elif default is False and type_ in {bool, None}: # flag
                parser.add_argument(
                    '--' + name, default=False, action='store_true'
                )
            else: # optional argument
                if type_ is None and default is not None:
                    type_ = type(default)
                parser.add_argument(
                    '--' + name, default=default, type=type_, help=f'[{default}]'
                )

        # parse and display command line arguments
        kwargs = vars(parser.parse_args(sys.argv[1:]))
        print(kwargs)

        # call the main function
        func(**kwargs)

    return func

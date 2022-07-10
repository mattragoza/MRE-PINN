import numpy as np

from .utils import copy_metadata

grad = np.gradient


def jacobian(u, resolution, i, j):
    return grad(u[...,i], axis=j) / resolution


def hessian(u, resolution, component, i, j):
    return grad(grad(u[...,component], axis=i), axis=j) / resolution**2


@copy_metadata
def laplacian(u, resolution=1, dim=0):
    '''
    Discrete Laplacian operator.

    Args:
        u: (..., M) output tensor.
        resolution: Input resolution.
        dim: Summation start axis.
    Returns:
        L: (..., M) Laplacian tensor.
    '''
    components = []
    for i in range(u.shape[-1]):
        component = 0
        for j in range(dim, u.ndim - 1):
            if u.shape[j] > 1:
                component += hessian(u, resolution, component=i, i=j, j=j)
        components.append(component)
    return np.stack(components, axis=-1)


def helmholtz_inversion(u, Lu, omega, rho=1000):
    '''
    Direct algebraic inversion
    of the Helmholtz equation.
    '''
    axes = tuple(range(1, u.ndim))
    omega = np.expand_dims(omega, axis=axes)
    return (-rho * (2 * np.pi * omega)**2 * u / Lu).mean(axis=-1)


@copy_metadata
def sfft(u, shift=True):
    '''
    Convert to spatial frequency domain.
    '''
    axes = tuple(range(1, u.ndim))
    u_f = np.fft.fftn(u, axes=axes)
    if shift:
        return np.fft.fftshift(u_f, axes=axes)
    return u_f

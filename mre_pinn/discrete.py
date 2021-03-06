import numpy as np
import xarray as xr

from .utils import copy_metadata


def grad(u, dim):
    '''
    Finite difference gradient.

    Args:
        u: An xarray of function values.
        dim: Coordinate to differentiate with
            respect to, using finite differences.
    Returns:
        du: An xarray of derivative values.
    '''
    return u.differentiate(coord=dim)


def jacobian(u, i, j):
    '''
    Evaluate discrete Jacobian matrix.

    Args:
        u: An xarray of vector-valued function
            evaluations, which has a component dim
            representing the vector components.
        i: Vector component to differentiate.
        j: Coordinate to differentate with respect to.
    Returns:
        du_i/dx_j
    '''
    return grad(u.sel(component=i), dim=j)


def hessian(u, component, i, j):
    '''
    Evaluate discrete Hessian matrix.

    Args:
        u: An xarray of vector-valued function
            evaluations, which has a component dim
            representing the vector components.
        component: Vector component to differentiate.
        i: First coordinate to differentate with respect to.
        j: Second coordinate to differentate with respect to.
    Returns:
        du_component/dx_ij
    '''
    return grad(grad(u.sel(component=component), dim=i), dim=j)


def laplacian(u):
    '''
    Evaluate discrete Laplacian operator.

    Args:
        u: (..., M) xarray of function values.
    Returns:
        Lu: (..., M) xarray of Laplacian values.
    '''
    components = []
    for idx, i in enumerate(u.component):
        component = 0
        for j in u.field.spatial_dims:
            component += hessian(u, component=i, i=j, j=j)
        components.append(component)
    return xr.concat(components, dim=u.component).transpose(*u.dims)


def helmholtz_inversion(u, Lu, rho=1000):
    '''
    Direct algebraic inversion
    of the Helmholtz equation.
    '''
    axes = tuple(range(1, u.ndim))
    omega = np.expand_dims(u.frequency, axis=axes)
    return (-rho * (2 * np.pi * omega)**2 * u / Lu).mean(axis=-1)


@copy_metadata
def fft(u, shift=True):
    '''
    Convert to spatial frequency domain.
    '''
    axes = u.field.spatial_axes
    u_f = np.fft.fftn(u, norm='ortho', axes=axes)
    if shift:
        return np.fft.fftshift(u_f, axes=axes)
    return u_f


def power_spectrum(u, n_bins=10):
    '''
    Compute power density wrt spatial frequency.
    '''
    # compute power spectrum
    ps = np.abs(fft(u))**2
    ps.name = u.name

    # compute spatial frequency radii for binning
    x = ps.field.spatial_points(reshape=False, standardize=True)
    r = np.linalg.norm(x, ord=2, axis=-1)
    ps = ps.assign_coords(spatial_frequency=(ps.field.spatial_dims, r * n_bins))

    # take mean across spatial frequency bins
    bins = np.linspace(0, n_bins, n_bins + 1, endpoint=True)
    ps = ps.groupby_bins('spatial_frequency', bins=bins).mean(...)
    return ps #.values


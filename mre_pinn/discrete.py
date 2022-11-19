import numpy as np
import xarray as xr
import scipy.special
from functools import cache

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
    try:
        return u.differentiate(coord=dim)
    except IndexError:
        # singleton dim has no derivative
        return xr.zeros_like(u)


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


def helmholtz_inversion(
    u, Lu, rho=1000, frequency=None, polar=False, eps=1e-8
):
    '''
    Direct algebraic inversion of the Helmholtz equation.

    Args:
        u: An xarray of wave field values.
        Lu: An xarray of Laplacian values.
        rho: Material density parameter.
        polar: Use polar complex representation.
        eps: Numerical parameter.
    Returns:
        An xarray of shear modulus values.
    '''
    if frequency is None: # use frequency coordinate
        omega = 2 * np.pi * u.frequency
        omega = np.expand_dims(omega, axis=tuple(range(1, u.ndim)))
    else:
        omega = 2 * np.pi * frequency

    if polar and np.iscomplexobj(u):
        numer_abs_G = (rho * omega**2 * np.abs(u))
        denom_abs_G = np.abs(Lu)
        numer_phi_G = (u.real * Lu.real + u.imag * Lu.imag)
        denom_phi_G = (np.abs(u) * np.abs(Lu))

        if u.field.has_components: # vector field
            numer_abs_G = numer_abs_G.sum(axis=-1)
            denom_abs_G = denom_abs_G.sum(axis=-1)
            numer_phi_G = numer_phi_G.sum(axis=-1)
            denom_phi_G = denom_phi_G.sum(axis=-1)

        abs_G = numer_abs_G / (denom_abs_G + eps)
        phi_G = np.arccos(-numer_phi_G / (denom_phi_G + eps))
        return abs_G * np.exp(1j * phi_G)
    else:
        numer_mu = (-rho * omega**2 * u)
        denom_mu = Lu

        if u.field.has_components: # vector field
            numer_mu = numer_mu.sum(axis=-1)
            denom_mu = denom_mu.sum(axis=-1)

        return numer_mu / (denom_mu + eps)


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


@cache
def savgol_filter_nd(n, order=1, kernel_size=3):
    '''
    N-dimensional Savitsky-Golay filter.

    This function creates a set of convolutional kernels
    that, when applied to an array of data values, are
    equivalent to fitting polynomials to local windows of
    data and then evaluating them or their derivatives.

    Args:
        n: Number of spatial dimensions.
        order: Order of polynomials to fit to each window.
        kernel_size: Size of windows to fit with polynomials.
    Returns:
        A dict mapping from derivative orders (n-tuples of ints)
            to conv kernels (numpy arrays of size kernel_size^n).
    '''
    assert kernel_size % 2 == 1, 'kernel_size must be odd'
    
    # relative coordinates of kernel values
    half_size = kernel_size // 2
    coords = np.arange(-half_size, half_size + 1)
    coords = np.stack(np.meshgrid(*[coords] * n), axis=-1).reshape(-1, n)
    n_values = len(coords)
    
    # powers of polynomial terms
    powers = np.arange(order + 1)
    powers = np.stack(np.meshgrid(*[powers] * n), axis=-1).reshape(-1, n)
    powers = powers[powers.sum(axis=1) <= order]
    n_terms = len(powers)
    
    assert n_values >= n_terms, 'order is too high for kernel_size'
    
    # set up linear system of equations
    A = np.zeros((n_values, n_terms))
    for i in range(n_values):
        for j in range(n_terms):
            A[i,j] = np.power(coords[i], powers[j]).prod()
    
    # compute the pseudo-inverse of the coefficient matrix
    kernels = np.linalg.pinv(A)
    
    # this factor is needed for correct derivative kernels
    kernels *= scipy.special.factorial(powers).prod(axis=1, keepdims=True)

    kernel_shape = (kernel_size,) * n
    kernels = kernels.reshape(-1, *kernel_shape)
    
    # return mapping from derivative order to kernel
    return {tuple(p): k for p, k in zip(powers, kernels)}


def savgol_smoothing(u, **kwargs):
    ndim = u.field.n_spatial_dims
    deriv_order = (0,) * ndim
    kernels = savgol_filter_nd(ndim, **kwargs)
    return scipy.ndimage.convolve(u, kernels[deriv_order], mode='reflect')


def savgol_jacobian(u, **kwargs):
    ndim = u.field.n_spatial_dims
    resolution = u.field.spatial_resolution
    kernels = savgol_filter_nd(ndim, **kwargs)
    new_dim = u.component.rename(component='gradient')
    Ju = u.expand_dims({'gradient': new_dim}, axis=-1).copy()
    for frequency in range(u.shape[0]):
        for component in range(u.shape[-1]):
            for i, deriv_order in enumerate(np.eye(ndim, dtype=int)):
                deriv_order = tuple(deriv_order)
                Ju[frequency,...,component,i] = scipy.ndimage.convolve(
                    u[frequency,...,component], kernels[deriv_order], mode='reflect'
                ) / resolution[i]
    return Ju


def savgol_laplacian(u, **kwargs):
    ndim =  u.field.n_spatial_dims
    resolution = u.field.spatial_resolution
    kernels = savgol_filter_nd(u.ndim, **kwargs)
    Lu = xr.zeros_like(u)
    for i, deriv_order in enumerate(2 * np.eye(ndim, dtype=int)):
        deriv_order = tuple(deriv_order)
        Lu += scipy.ndimage.convolve(
            u, kernels[deriv_order], mode='reflect'
        ) / resolution[i]**2
    return Lu

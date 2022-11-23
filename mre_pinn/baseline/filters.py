from functools import cache
import numpy as np
import scipy.ndimage
import scipy.special

from ..utils import copy_metadata


@cache
def savgol_kernel_nd(n, order=3, kernel_size=5):
    '''
    N-dimensional Savitsky-Golay filter kernel.

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


@copy_metadata
def outlier_filter(a, threshold):
    if np.iscomplexobj(a):
        abs_v = outlier_filter(np.abs(a), threshold)
        angle = outlier_filter(np.angle(a), threshold)
        return abs_v * np.exp(1j * angle)
    k = np.array([1, 2, 3, 2, 1])
    k = np.einsum('i,j->ij', k, k)
    k = k > 1
    if a.ndim == 4:
        k = k[...,None,None]
    elif a.ndim == 3:
        k = k[...,None]
    a_median = scipy.ndimage.median_filter(a, footprint=k)
    a_diff = (a - a_median) / a_median
    a = np.where(a_diff >  threshold,  threshold * a_median, a)
    a = np.where(a_diff < -threshold, -threshold * a_median, a)
    return a


gaussian_filter = copy_metadata(scipy.ndimage.gaussian_filter)

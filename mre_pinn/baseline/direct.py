import numpy as np
import xarray as xr

from . import filters


def eval_direct_baseline(
    example,
    frequency,
    polar=True,
    savgol_filter=True,
    order=3,
    kernel_size=5,
    despeckle='auto',
    threshold=3
):
    print('Evaluating direct baseline')
    u = example.wave
    if savgol_filter:
        u = u.field.smooth(order=order, kernel_size=kernel_size)
        Lu = u.field.laplacian(order=order, kernel_size=kernel_size, savgol=True)
    else:
        Lu = u.field.laplacian(savgol=False)

    if despeckle:
        Lu = 1 / filters.outlier_filter(1 / Lu, threshold=threshold)

    Mu = helmholtz_inversion(u, Lu, frequency=frequency, polar=polar)
    Mu.name = 'baseline'
    example['direct'] = Mu


def helmholtz_inversion(
    u, Lu, rho=1000, frequency=None, polar=False, eps=1e-8
):
    '''
    Algebraic direct inversion of the Helmholtz equation.

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

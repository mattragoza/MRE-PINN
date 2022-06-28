import numpy as np

grad = np.gradient


def jacobian(u, resolution, i, j):
    return grad(u[...,i], axis=j) / resolution


def hessian(u, resolution, component, i, j):
    return grad(grad(u[...,component], axis=i), axis=j) / resolution**2


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
            component += hessian(u, resolution, component=i, i=j, j=j)
        components.append(component)
    return np.stack(components, axis=-1)


def helmholtz_inversion(u, Lu, omega, rho=1000):
    '''
    Direct algebraic inversion
    of the Helmholtz equation.
    '''
    return (-rho * (2 * np.pi * omega)**2 * u / Lu).mean(axis=-1)

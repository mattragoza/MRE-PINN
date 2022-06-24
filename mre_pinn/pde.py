import numpy as np
import torch
import deepxde


def laplacian(u, x, dim=0):
    components = []
    for i in range(u.shape[1]):
        component = 0
        for j in range(dim, x.shape[1]):
            component += deepxde.grad.hessian(u, x, component=i, i=j, j=j)
        components.append(component)
    return torch.cat(components, dim=1)


class WaveEquation(object):

    def __init__(self, detach, rho=1000, dx=1):
        self.detach = detach
        self.rho = rho
        self.dx  = dx

    def __call__(self, x, outputs):
        u, mu = outputs[:,:-1], outputs[:,-1:]
        omega = x[:,:1]
        lu = laplacian(u, x, dim=1) / self.dx**2
        if self.detach:
            u, lu = u.detach(), lu.detach()
        return mu * lu + self.rho * (2*np.pi*omega)**2 * u


def lvwe(x, u, mu, lam, rho, omega):
    '''
    General form of the steady-state
    linear viscoelastic wave equation.
    '''
    jac_u = jacobian(u, x)

    strain = (1/2) * (jac_u + jac_u.T)
    stress = 2 * mu * strain + lam * trace(strain) * I

    lhs = divergence(stress, x)

    return lhs + rho * omega**2 * u


def homogeneous_lvwe(x, u, mu, lam, rho, omega):
    '''
    Linear viscoelastic wave equation
    with assumption of homogeneity.
    '''
    laplace_u = laplacian(u, x)
    div_u = divergence(u, x)
    grad_div_u = gradient(div_u, x)

    lhs = mu * laplace_u + (lam + mu) * grad_div_u

    return lhs + rho * omega**2 * u


def incompressible_homogeneous_lvwe(x, u, mu, rho, omega):
    '''
    Linear viscoelastic wave equation
    with assumption of homogeneity and
    incompressibility.
    '''
    laplace_u = laplacian(u, x)

    lhs = mu * laplace_u

    return lhs + rho * omega**2 * u


def pressure_homogeneous_lvwe(x, u, mu, p, rho, omega):
    '''
    Linear viscoelastic wave equation
    with assumption of homogeneity and
    additional pressure term.
    '''
    laplace_u = laplacian(u, x)
    grad_p = gradient(p, x)

    lhs = mu * laplace_u + grad_p

    return lhs + rho * omega**2 * u

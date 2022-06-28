import numpy as np
import torch
import deepxde

hessian = deepxde.grad.hessian


def laplacian(u, x, dim=0):
    '''
    Continuous Laplacian operator.

    Args:
        u: (N, M) output tensor.
        x: (N, K) input tensor.
        dim: Summation start index.
    Returns:
        v: (N, M) Laplacian tensor.
    '''
    components = []
    for i in range(u.shape[1]):
        component = 0
        for j in range(dim, x.shape[1]):
            component += hessian(u, x, component=i, i=j, j=j)
        components.append(component)
    return torch.cat(components, dim=1)


class WaveEquation(object):
    '''
    ∇·σ + ρω²u = 0
    '''
    def __init__(
        self, detach, rho=1000, dx=1, homogeneous=False, debug=False
    ):
        self.detach = detach
        self.rho = rho
        self.dx = dx
        self.homogeneous = homogeneous
        self.incompressible = True
        self.debug = debug

    def __call__(self, x, outputs):
        '''
        Args:
            x: (N x 4) input tensor of omega,x,y,z
            outputs: (N x 4) tensor of ux,uy,uz,mu
        Returns:
            (N x 3) tensor of PDE residual for each
                ux,uy,uz displacement component
        '''
        if self.debug:
            u, lu = outputs[:,0:2], outputs[:,2:]
            laplace_u = laplacian(u, x, dim=1) / self.dx**2
            return laplace_u.detach() - lu

        u, mu = outputs[:,0:-1], outputs[:,-1:]
        omega = x[:,:1]
        f = self.rho * (2 * np.pi * omega)**2 * u

        if self.homogeneous:

            # Helmholtz equation
            laplace_u = laplacian(u, x, dim=1) / self.dx**2
            div_stress = mu * laplace_u
        else:
            # Barnhill 2017
            div_stress = mu * 0
        
        if self.detach: # only backprop through mu
            u, laplace_u = u.detach(), laplace_u.detach()

        return div_stress + f


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

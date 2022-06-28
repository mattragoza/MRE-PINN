import numpy as np
import torch
import deepxde


def jacobian(u, x, dim=0):
    '''
    Args:
        u: (N, M) output tensor.
        x: (N, K) input tensor.
    Returns:
        J: (N, M, K) Jacobian tensor.
    '''
    components = []
    for i in range(u.shape[1]):
        component = deepxde.grad.jacobian(u, x, i)[:,dim:]
        components.append(component)
    return torch.stack(components, dim=1)


def divergence(u, x, dim=0):
    '''
    Trace of the Jacobian matrix.

    Args:
        y: (N, M, K) output tensor.
        x: (N, K) input tensor.
    Returns:
        div: (N, M) divergence tensor.
    '''
    components = []
    for i in range(u.shape[1]):
        component = 0
        for j in range(u.shape[2]):
            component += deepxde.grad.jacobian(u[:,i], x, j, dim+j)
        components.append(component)
    return torch.cat(components, dim=1)


def laplacian(u, x, dim=0):
    '''
    Continuous Laplacian operator.

    Args:
        u: (N, M) output tensor.
        x: (N, K) input tensor.
        dim: Summation start index.
    Returns:
        L: (N, M) Laplacian tensor.
    '''
    components = []
    for i in range(u.shape[1]):
        i = i if u.shape[1] > 1 else None
        component = 0
        for j in range(dim, x.shape[1]):
            component += deepxde.grad.hessian(u, x, i, j, j)
        components.append(component)
    return torch.cat(components, dim=1)


class WaveEquation(object):
    '''
    Navier-Cauchy equation for steady-state
    elastic wave vibration.

    ∇·[μ(∇u + (∇u)ᵀ) + λ(∇·u)I] = -ρω²u
    '''
    def __init__(
        self, detach, rho=1000,
        homogeneous=True,
        incompressible=True,
        debug=False
    ):
        self.detach = detach
        self.rho = rho
        self.homogeneous = homogeneous
        self.incompressible = incompressible
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
        u, mu = outputs[:,:-1], outputs[:,-1:]
        omega = x[:,:1]

        if self.homogeneous:

            laplace_u = laplacian(u, x, dim=1)
            if self.detach:
                laplace_u = laplace_u.detach()

            if self.incompressible: # Helmholtz
                div_stress = mu * laplace_u
            else:
                div_u = divergence(u.unsqueeze(1), x, dim=1)
                grad_div_u = jacobian(div_u, x, dim=1)[:,0,:]
                if self.detach:
                    grad_div_u = grad_div_u.detach()

                div_stress = mu * laplace_u + mu * grad_div_u

        else: 
            if self.incompressible: # Barnhill 2017

                grad_u = jacobian(u, x, dim=1)
                grad_u += torch.transpose(grad_u, 1, 2)
                div_grad_u = divergence(grad_u, x, dim=1)
                #print(grad_u.shape, div_grad_u.shape)

                if self.detach:
                    grad_u = grad_u.detach()
                    div_grad_u = div_grad_u.detach()

                grad_mu = jacobian(mu, x, dim=1)
                #print(mu.shape, grad_mu.shape)
                
                div_stress = mu * div_grad_u + (grad_mu * grad_u).sum(dim=1)

        if self.detach:
            u = u.detach()
        
        f = self.rho * (2 * np.pi * omega)**2 * u
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

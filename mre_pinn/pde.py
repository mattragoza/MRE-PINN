import numpy as np
import torch
import deepxde
from functools import wraps


class WaveEquation(object):
    '''
    Navier-Cauchy equation for steady-state
    elastic wave vibration.

    ∇·[μ(∇u + (∇u)ᵀ) + λ(∇·u)I] = -ρω²u

    Args:
        homogeneous: Assume ∇μ = ∇λ = 0
        incompressible: Assume ∇·u = 0
        rho: Material density in kg/m³
        lambda_: Lame's first parameter
        detach: Do not backprop through u
    '''
    @classmethod
    def from_name(cls, pde_name, **kwargs):
        pde_name = pde_name.lower()
        if pde_name == 'helmholtz':
            return HelmholtzEquation(**kwargs)
        elif pde_name == 'hetero':
            return HeteroEquation(**kwargs)
        elif pde_name == 'compress':
            return CompressEquation(**kwargs)
        elif pde_name == 'general':
            return WaveEquation(**kwargs)
        elif pde_name == 'debug':
            return DebugEquation(**kwargs)
        else:
            raise ValueError(f'unrecognized PDE name: {pde_name}')

    def __init__(self, rho=1000, lambda_=0, detach=True):
        self.rho, self.lambda_ = rho, lambda_
        self.detach = detach

    def traction_forces(self, x, u, mu):

        grad_u = jacobian(u, x, dim=1)
        if self.detach:
            grad_u = grad_u.detach()

        strain = (1/2) * (grad_u + grad_u.transpose(1, 2))
        mu_strain = mu.unsqueeze(-1) * 2 * strain

        tr_strain = strain.diagonal(1, 2).sum(dim=1, keepdims=True)
        I = torch.eye(u.shape[1]).unsqueeze(0)
        lambda_strain = self.lambda_ * tr_strain.unsqueeze(-1) * I

        stress = mu_strain + lambda_strain

        # is this divergence correct if we detach u?
        return divergence(stress, x, dim=1)

    def body_forces(self, omega, u):
        if self.detach:
            u = u.detach()
        return self.rho * omega**2 * u

    def traction_and_body_forces(self, x, u, mu):
        '''
        Compute the traction and body forces.

        Args:
            x: (N x D + 1) input tensor of frequency
                and spatial coordinates
            u: (N x D) tensor of displacement vectors
            mu: (N x 1) tensor of shear modulus
        Returns:
            2 (N x D) tensors containing the traction and
                body forces for each displacement component
        '''
        omega = 2 * np.pi * x[:,:1] # radians
        f_trac = self.traction_forces(x, u, mu)
        f_body = self.body_forces(omega, u)
        return f_trac, f_body

    def __call__(self, x, u, mu):
        '''
        Compute the PDE residuals.

        Args:
            x: (N x D + 1) input tensor of frequency
                and spatial coordinates
            u: (N x D) tensor of displacement vectors
            mu: (N x 1) tensor of shear modulus
        Returns:
            (N x D) tensor of PDE residual for each
                displacement component
        '''
        f_trac, f_body = self.traction_and_body_forces(x, u, mu)
        return f_trac + f_body


class HelmholtzEquation(WaveEquation):

    def traction_forces(self, x, u, mu):
        laplace_u = laplacian(u, x, dim=1)
        if self.detach:
            laplace_u = laplace_u.detach()
        return mu * laplace_u


class HeteroEquation(WaveEquation):

    def traction_forces(self, x, u, mu):

        grad_u = jacobian(u, x, dim=1)

        # doesn't ∇·u = 0 ⟹ ∇(∇·u) = 0?
        # meaning we don't need the next term
        # and div_grad_u is just laplace_u
        #grad_u = (grad_u + grad_u.transpose(1, 2))

        div_grad_u = divergence(grad_u, x, dim=1)

        if self.detach:
            grad_u = grad_u.detach()
            div_grad_u = div_grad_u.detach()

        grad_mu = jacobian(mu, x, dim=1)

        return mu * div_grad_u + (grad_mu * grad_u).sum(dim=1)


class CompressEquation(WaveEquation):

    def traction_forces(self, x, u, mu):

        laplace_u = laplacian(u, x, dim=1)
        if self.detach:
            laplace_u = laplace_u.detach()

        # pressure term
        div_u = divergence(u.unsqueeze(1), x, dim=1)
        grad_div_u = jacobian(div_u, x, dim=1)[:,0,:]
        if self.detach:
            grad_div_u = grad_div_u.detach()

        return mu * laplace_u + (self.lambda_ + mu) * grad_div_u


class DebugEquation(WaveEquation):

    def body_forces(self, omega, u):
        return u * 0

    def traction_forces(self, x, u, mu):
        return u * 0


def complex_operator(f):
    @wraps(f)
    def wrapper(u, x, dim=0):
        if u.dtype.is_complex:
            real = f(u.real, x, dim)
            imag = f(u.imag, x, dim)
            return real + 1j * imag
        else:
            return f(u, x, dim)
    return wrapper


@complex_operator
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


@complex_operator
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


@complex_operator
def laplacian(u, x, dim=0):
    '''
    Continuous Laplacian operator,
    which is the divergence of the
    gradient.

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

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

    def __init__(self, rho=1000, omega=None, lambda_=0, detach=True):
        self.rho = rho         # mass density
        self.omega = omega     # time frequency
        self.lambda_ = lambda_ # Lame parameter
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
        omega = 2 * np.pi * self.omega #x[:,:1] # radians
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
        laplace_u = laplacian(u, x)
        if self.detach:
            laplace_u = laplace_u.detach()
        return mu * laplace_u


class HeteroEquation(WaveEquation):

    def traction_forces(self, x, u, mu):

        grad_u = jacobian(u, x)

        # doesn't ∇·u = 0 ⟹ ∇(∇·u) = 0?
        # meaning we don't need the next term
        # and div_grad_u is just laplace_u
        #grad_u = (grad_u + grad_u.transpose(1, 2))

        div_grad_u = divergence(grad_u, x)

        if self.detach:
            grad_u = grad_u.detach()
            div_grad_u = div_grad_u.detach()

        grad_mu = jacobian(mu, x)

        return mu * div_grad_u + (grad_mu * grad_u).sum(dim=-1)


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
    def wrapper(u, x, *args, **kwargs):
        if u.dtype.is_complex:
            f_real = f(u.real, x, *args, **kwargs)
            f_imag = f(u.imag, x, *args, **kwargs)
            return f_real + 1j * f_imag
        else:
            return f(u, x, *args, **kwargs)
    return wrapper


@complex_operator
def gradient(u, x, no_z=True):
    '''
    Continuous gradient operator, which maps a
    scalar field to a vector field of partial
    derivatives.

    Args:
        u: (..., 1) output tensor.
        x: (..., K) input tensor.
    Returns:
        D: (..., K) gradient tensor, where:
            D[...,i] = ∂u[...,0] / ∂x[...,i]
    '''
    assert u.shape[:-1] == x.shape[:-1]
    assert u.shape[-1] == 1
    if False:
        grad = deepxde.grad.jacobian(u.view(-1, 1), x.view(-1, x.shape[-1]))
        return grad.reshape(x.shape)
    else:
        ones = torch.ones_like(u)
        grad = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
        if no_z:
            return grad[...,:2]
        else:
            return grad


def jacobian(u, x, no_z=True):
    '''
    Continuous Jacobian operator. The Jacobian
    is the gradient operator for vector fields.
    It maps a vector field to a tensor field of
    partial derivatives.

    Args:
        u: (..., M) output tensor.
        x: (..., K) input tensor.
    Returns:
        J: (..., M, K) Jacobian tensor, where:
            J[...,i,j] = ∂u[...,i] / ∂x[...,j]
    '''
    components = []
    for i in range(u.shape[-1]):
        component = gradient(u[...,i:i+1], x, no_z)
        if no_z:
            component = component[...,:2]
        components.append(component)
    return torch.stack(components, dim=-2)


def divergence(u, x, no_z=True):
    '''
    Continuous tensor divergence operator.
    Divergence is the trace of the Jacobian,
    and maps a tensor field to vector field.

    Args:
        u: (..., M, K) output tensor.
        x: (..., K) input tensor.
    Returns:
        V: (..., M) divergence tensor, where:
            V[...,i] = ∑ⱼ ∂u[...,i,j] / ∂x[...,j]
    '''
    components = []
    for i in range(u.shape[-2]):
        J = jacobian(u[...,i,:], x, no_z)
        component =  0
        for j in range(J.shape[-1]):
            component += J[...,j,j]
        components.append(component)
    return torch.stack(components, dim=-1)


def laplacian(u, x, no_z=True):
    '''
    Continuous vector Laplacian operator.
    The Laplacian is the divergence of the
    gradient, and maps a vector field to
    a vector field.

    Args:
        u: (..., M) output tensor.
        x: (..., K) input tensor.
    Returns:
        L: (..., M) Laplacian tensor, where:
            L[...,i] = ∑ⱼ ∂²u[...,i] / ∂x[...,j]²
    '''
    return divergence(jacobian(u, x, no_z), x, no_z)

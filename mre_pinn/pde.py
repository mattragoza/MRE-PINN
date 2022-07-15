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
            return WaveEquation(homogeneous=True, incompressible=True, **kwargs)
        elif pde_name == 'hetero':
            return WaveEquation(homogeneous=False, incompressible=True, **kwargs)
        elif pde_name == 'compress':
            return WaveEquation(homogeneous=True, incompressible=False, **kwargs)
        elif pde_name == 'general':
            return WaveEquation(homogeneous=False, incompressible=False, **kwargs)
        else:
            raise ValueError(f'unrecognized PDE name: {pde_name}')

    def __init__(
        self,
        homogeneous,
        incompressible,
        rho=1000,
        lambda_=0,
        detach=True,
        debug=False
    ):
        self.homogeneous = homogeneous
        self.incompressible = incompressible
        self.rho, self.lambda_ = rho, lambda_
        self.detach = detach
        self.debug = debug

    def __call__(self, x, outputs):
        '''
        Compute the PDE residual.

        Args:
            x: (N x 4) input tensor of omega,x,y,z
            outputs: (N x 4) tensor of ux,uy,uz,mu
        Returns:
            (N x 3) tensor of PDE residual for each
                ux,uy,uz displacement component
        '''
        u, mu = outputs[:,:-1], outputs[:,-1:]
        omega = 2 * np.pi * x[:,:1] # radians

        if self.debug: # no PDE residual
            return u * 0

        elif self.homogeneous: # Helmholtz equation

            laplace_u = laplacian(u, x, dim=1)
            if self.detach:
                laplace_u = laplace_u.detach()

            div_stress = mu * laplace_u

            if not self.incompressible: # add pressure term

                div_u = divergence(u.unsqueeze(1), x, dim=1)
                grad_div_u = jacobian(div_u, x, dim=1)[:,0,:]
                if self.detach:
                    grad_div_u = grad_div_u.detach()

                div_stress += (self.lambda_ + mu) * grad_div_u

        elif self.incompressible: # Barnhill 2017

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
            div_stress = mu * div_grad_u + (grad_mu * grad_u).sum(dim=1)

        else: # general form

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
            div_stress = divergence(stress, x, dim=1)

        if self.detach:
            u = u.detach()

        return div_stress + self.rho * omega**2 * u

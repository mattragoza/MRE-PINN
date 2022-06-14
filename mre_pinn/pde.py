import torch
import deepxde


class HelmholtzPDE(object):

	def __init__(self, rho, omega):
		self.rho = rho
		self.omega = omega

	def __call__(self, x, outputs):
		u, mu = torch.split(outputs, [2, 1], dim=1)
		ux_xx = deepxde.grad.hessian(u, x, component=0, i=0, j=0)
		ux_yy = deepxde.grad.hessian(u, x, component=0, i=1, j=1)
		uy_xx = deepxde.grad.hessian(u, x, component=1, i=0, j=0)
		uy_yy = deepxde.grad.hessian(u, x, component=1, i=1, j=1)
		laplace_u = torch.cat([ux_xx + ux_yy, uy_xx + uy_yy], dim=1)
		return mu * laplace_u + self.rho * self.omega**2 * u


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

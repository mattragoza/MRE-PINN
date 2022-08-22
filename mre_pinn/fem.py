import numpy as np
import scipy.interpolate

# FENICSx imports
import ufl, dolfinx
from mpi4py import MPI
from petsc4py.PETSc import ScalarType as dtype


class FEM(object):

	def __init__(self, data):

		x = data.u.field.spatial_points()
		u = data.u.field.values()

		# discretize the domain on a mesh
		self.mesh = dolfinx.mesh.create_rectangle(
			comm=MPI.COMM_WORLD,
			points=[x.min(axis=0), x.max(axis=0)],
			n=data.u.field.spatial_shape,
			cell_type=dolfinx.mesh.CellType.triangle,
			diagonal=dolfinx.mesh.DiagonalType.right_left
		)

		# define the FEM basis function spaces
		ndim = data.field.n_spatial_dims
		self.scalar_space = dolfinx.fem.FunctionSpace(
			self.mesh, ('Lagrange', 1)
		)
		self.vector_space = dolfinx.fem.VectorFunctionSpace(
			self.mesh, ('Lagrange', 1), dim=ndim
		)
		self.tensor_space = dolfinx.fem.TensorFunctionSpace(
			self.mesh, ('Lagrange', 1), shape=(ndim, ndim)
		)

		# setup physical problem and function spaces
		self.rho = 1000
		self.omega = 2 * np.pi * data.frequency.values.item()

		print(x.shape, u.shape)

		# function for interpolating wave image into FEM basis
		u_interp = scipy.interpolate.LinearNDInterpolator(points=x, values=u)
		self.u_func = dolfinx.fem.Function(self.vector_space)

		def f(x):
			return np.ascontiguousarray(u_interp(x[:ndim].T).T)

		self.u_func.interpolate(f)

		# trial and test functions
		self.mu_func = ufl.TrialFunction(self.scalar_space)
		self.v_func = ufl.TestFunction(self.vector_space)
		self.mu_pred_func = None

	def solve(self):

		Ax = self.mu_func * ufl.inner(
			ufl.grad(self.u_func), ufl.grad(self.v_func)
		) * ufl.dx

		b = self.rho * self.omega**2 * ufl.inner(
			self.u_func, self.v_func
		) * ufl.dx

		problem = dolfinx.fem.petsc.LinearProblem(
			Ax, b, bcs=[], petsc_options={'ksp_type': 'lsqr', 'pc_type': 'none'}
		)
		self.mu_pred_func = problem.solve()

	def __call__(self, x):
		return eval_dolfinx_func(self.mu_pred_func, x)


def get_containing_cells(mesh, x):
	'''
	Get indices of mesh cells that contain the given points.
	'''
	tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
	cells = dolfinx.geometry.compute_collisions(tree, x)
	cells = dolfinx.geometry.compute_colliding_cells(mesh, cells, x)
	return [cells.links(i)[0] for i in range(len(x))]


def eval_dolfinx_func(f, x):
	'''
	Evaluate a dolfinx function on a set of points.

	Args:
		f: A dolfinx FEM function.
		x: A numpy array of points.
	Returns:
		Numpy array of f evaluated at x.
	'''
	x = np.concatenate([
		x, np.zeros((x.shape[0], 3 - x.shape[1]))
	], axis=1)
	cells = get_containing_cells(f.function_space.mesh, x)
	return f.eval(x, cells)

import numpy as np
import scipy.interpolate

# FENICSx imports
import ufl, dolfinx
from mpi4py import MPI
from petsc4py.PETSc import ScalarType as dtype

from .utils import as_xarray


class FEM(object):

    def __init__(self, data, n_mesh=None, extend=0):

        self.data = data
        ndim = data.field.n_spatial_dims
        x = data.u.field.spatial_points()
        u = data.u.field.values()
        x_res = x[1] - x[0] if extend else 0
        print(x_res, data.u.field.spatial_shape)

        # discretize the domain on a mesh
        if ndim == 2:
            self.mesh = dolfinx.mesh.create_rectangle(
                comm=MPI.COMM_WORLD,
                points=[x.min(axis=0) - extend, x.max(axis=0) + extend],
                n=data.u.field.spatial_shape,
                cell_type=dolfinx.mesh.CellType.triangle,
                diagonal=dolfinx.mesh.DiagonalType.right_left
            )
        elif ndim == 1:
            self.mesh = dolfinx.mesh.create_interval(
                comm=MPI.COMM_WORLD,
                points=[x.min(axis=0) - extend, x.max(axis=0) + extend],
                nx=n_mesh or data.u.field.spatial_shape[0]
            )

        # define the FEM basis function spaces
        self.scalar_space = dolfinx.fem.FunctionSpace(
            self.mesh, ('CG', 1)
        )
        self.vector_space = dolfinx.fem.VectorFunctionSpace(
            self.mesh, ('CG', 1), dim=ndim
        )
        self.tensor_space = dolfinx.fem.TensorFunctionSpace(
            self.mesh, ('CG', 1), shape=(ndim, ndim)
        )

        # setup physical problem and function spaces
        self.rho = 1000
        self.omega = 2 * np.pi * data.frequency.values.item()

        # function for interpolating wave image into FEM basis
        if ndim > 1:
            u_interp = scipy.interpolate.LinearNDInterpolator(x, u)
        else:
            u_interp = scipy.interpolate.interp1d(
                x[:,0], u[:,0], bounds_error=False, fill_value='extrapolate'
            )
        self.u_interp = u_interp
        self.u_func = dolfinx.fem.Function(self.vector_space)

        def f(x):
            return np.ascontiguousarray(u_interp(x[:ndim].T).T)

        self.u_func.interpolate(f)

        # trial and test functions
        self.mu_func = ufl.TrialFunction(self.scalar_space)
        self.v_func = ufl.TestFunction(self.vector_space)
        self.mu_pred_func = None

    def solve(self):

        # construct bilinear form
        Auv = self.mu_func * ufl.inner(
            ufl.grad(self.u_func), ufl.grad(self.v_func)
        ) * ufl.dx

        # construct inner product
        bv = self.rho * self.omega**2 * ufl.inner(
            self.u_func, self.v_func
        ) * ufl.dx

        # define the variational problem
        problem = dolfinx.fem.petsc.LinearProblem(
            Auv, bv, bcs=[], petsc_options={'ksp_type': 'lsqr', 'pc_type': 'none'}
        )
        self.mu_pred_func = problem.solve()

        x = self.data.u.field.spatial_points()
        mu_pred = eval_dolfinx_func(self.mu_pred_func, x)
        mu_pred = mu_pred.reshape(*self.data.mu.shape)
        mu_pred = as_xarray(mu_pred, like=self.data.mu)

        u_pred = eval_dolfinx_func(self.u_func, x)
        u_pred = u_pred.reshape(*self.data.u.shape)
        u_pred = as_xarray(u_pred, like=self.data.u)
        return u_pred, mu_pred


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

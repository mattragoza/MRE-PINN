import numpy as np
import scipy.interpolate

# FENICSx imports
import ufl, dolfinx
from mpi4py import MPI
from petsc4py.PETSc import ScalarType as dtype

from .utils import as_xarray


def parse_elem_type(s):
    family, degree = s.split('-')
    return family, int(degree)


def create_mesh_from_data(data, align_nodes):
    '''
    Create a mesh from the provided xarray data.

    Args:
        data: An xarray with spatial dims.
        align_nodes: If True, align the mesh nodes
            to the data points, otherwise align
            the mesh cells to the data points.
    Returns:
        A dolfinx mesh.
    '''
    # spatial dimensionality, points, and bounds
    ndim = data.field.n_spatial_dims
    x = data.field.spatial_points()
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    x_res = x[1] - x[0]

    if align_nodes: # align nodes to data
        bounds = [x_min, x_max]
        n_cells = [d - 1 for d in data.field.spatial_shape]

    else: # align cells to data
       bounds = [x_min - x_res/2, x_max + x_res/2]
       n_cells = data.field.spatial_shape

    # discretize the domain on a mesh
    if ndim == 2:
        mesh = dolfinx.mesh.create_rectangle(
            comm=MPI.COMM_WORLD,
            points=bounds,
            n=n_cells,
            cell_type=dolfinx.mesh.CellType.triangle,
            diagonal=dolfinx.mesh.DiagonalType.right_left
        )
    elif ndim == 1:
        mesh = dolfinx.mesh.create_interval(
            comm=MPI.COMM_WORLD,
            points=bounds,
            nx=n_cells[0]
        )
    return mesh


def create_func_from_data(data):

    ndim = data.field.n_spatial_dims
    x = data.field.spatial_points()
    y = data.field.values()

    if ndim > 1:
        interp = scipy.interpolate.LinearNDInterpolator(x, y)
    else:
        interp = scipy.interpolate.interp1d(
            x[:,0], y[:,0], bounds_error=False, fill_value='extrapolate'
        )
    def f(x):
        return np.ascontiguousarray(interp(x[:ndim].T).T)
    return f


class FEM(object):

    def __init__(
        self,
        data,
        u_elem_type='CG-1',
        mu_elem_type='DG-0',
        align_nodes=False
    ):
        self.data = data
        self.mesh = create_mesh_from_data(data, align_nodes)

        # determine the FEM element types
        self.u_elem_type = parse_elem_type(u_elem_type)
        self.mu_elem_type = parse_elem_type(mu_elem_type)

        # define the FEM basis function spaces
        ndim = data.field.n_spatial_dims
        self.scalar_space = dolfinx.fem.FunctionSpace(
            self.mesh, self.mu_elem_type
        )
        self.vector_space = dolfinx.fem.VectorFunctionSpace(
            self.mesh, self.u_elem_type, dim=ndim
        )
        self.tensor_space = dolfinx.fem.TensorFunctionSpace(
            self.mesh, self.u_elem_type, shape=(ndim, ndim)
        )

        # wave image-interpolating function
        self.u_func = dolfinx.fem.Function(self.vector_space)
        self.u_func.interpolate(create_func_from_data(data.u))

        # trial and test functions
        self.mu_func = ufl.TrialFunction(self.scalar_space)
        self.v_func = ufl.TestFunction(self.vector_space)

    def solve(self):

        # physical constants
        rho = 1000
        omega = 2 * np.pi * self.data.frequency.values.item()

        # construct bilinear form
        Auv = self.mu_func * ufl.inner(
            ufl.grad(self.u_func), ufl.grad(self.v_func)
        ) * ufl.dx

        # construct inner product
        bv = rho * omega**2 * ufl.inner(
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

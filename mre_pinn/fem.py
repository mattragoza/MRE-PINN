import numpy as np
import xarray as xr
import scipy.ndimage
import scipy.interpolate

# FENICSx imports
import ufl, dolfinx
from mpi4py import MPI
from petsc4py.PETSc import ScalarType as dtype

from .utils import as_xarray
from .discrete import savgol_filter_nd


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
    shape = np.array(data.field.spatial_shape)
    x = data.field.spatial_points()
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    x_res = (x_max - x_min) / (shape - 1)

    if align_nodes: # align nodes to data
        bounds = [x_min, x_max]
        n_cells = [d - 1 for d in data.field.spatial_shape]

    else: # align cells to data
        bounds = [x_min - x_res/2, x_max + x_res/2]
        n_cells = data.field.spatial_shape

    # discretize the domain on a mesh
    if ndim == 3:
        raise NotImplementedError

    elif ndim == 2:
        mesh = dolfinx.mesh.create_rectangle(
            comm=MPI.COMM_WORLD,
            points=bounds,
            n=n_cells,
            cell_type=dolfinx.mesh.CellType.triangle,
            diagonal=dolfinx.mesh.DiagonalType.left_right
        )
    elif ndim == 1:
        mesh = dolfinx.mesh.create_interval(
            comm=MPI.COMM_WORLD,
            points=bounds,
            nx=n_cells[0]
        )
    return mesh


def create_func_from_data(data):
    '''
    Create a function that linearly linterpolates
    between the values of the provided xarray data,
    assuming they are defined on a spatial grid.

    Args:
        data: An xarray with spatial dims.
    Returns:
        A function that linearly interpolates
            between the data values.
    '''
    ndim = data.field.n_spatial_dims
    x = data.field.spatial_points()
    y = data.field.values()

    if ndim > 1:
        interp = scipy.interpolate.LinearNDInterpolator(x, y)
    else:
        interp = scipy.interpolate.interp1d(
            x[:,0], y[:,0], bounds_error=False, fill_value='extrapolate'
        )
    def func(x):
        return np.ascontiguousarray(interp(x[:ndim].T).T)
    return func


class FEM(object):
    '''
    Solve Navier-Cauchy equation for steady-state
    elastic wave vibration using finite elements.
    '''
    def __init__(
        self,
        data,
        u_elem_type='CG-2',
        mu_elem_type='DG-0',
        align_nodes=True,
        savgol_filter=False,
    ):
        self.data = data
        self.mesh = create_mesh_from_data(data, align_nodes)

        ndim = data.field.n_spatial_dims
        shape = np.array(data.field.spatial_shape)
        x = data.field.spatial_points()
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
        x_res = (x_max - x_min) / (shape - 1)

        # determine the FEM element types
        self.u_elem_type = parse_elem_type(u_elem_type)
        self.mu_elem_type = parse_elem_type(mu_elem_type)

        # define the FEM basis function spaces
        self.scalar_space = dolfinx.fem.FunctionSpace(
            self.mesh, self.mu_elem_type
        )
        self.vector_space = dolfinx.fem.VectorFunctionSpace(
            self.mesh, self.u_elem_type, dim=ndim
        )
        self.tensor_space = dolfinx.fem.TensorFunctionSpace(
            self.mesh, self.u_elem_type, shape=(ndim, ndim)
        )

        if savgol_filter:

            # construct kernels for Savitzky-Golay filtering
            kernels = savgol_filter_nd(ndim, order=2, kernel_size=3)

            u = xr.zeros_like(data.u)
            for component in range(ndim):
                deriv_order = (0,) * ndim
                u[0,...,component] = scipy.ndimage.convolve(
                    data.u[0,...,component], kernels[deriv_order], mode='wrap'
                )

            # wave image-interpolating function
            self.u_func = dolfinx.fem.Function(self.vector_space)
            self.u_func.interpolate(create_func_from_data(u))

            Lu = xr.zeros_like(data.u)
            for component in range(ndim):
                for i, deriv_order in enumerate(2 * np.eye(ndim, dtype=int)):
                    deriv_order = tuple(deriv_order)
                    Lu[0,...,component] += scipy.ndimage.convolve(
                        data.u[0,...,component], kernels[deriv_order], mode='wrap'
                    ) / x_res[i]**2

            # Laplacian-interpolating function
            self.Lu_func = dolfinx.fem.Function(self.vector_space)
            self.Lu_func.interpolate(create_func_from_data(Lu))

        else:

            # wave image-interpolating function
            self.u_func = dolfinx.fem.Function(self.vector_space)
            self.u_func.interpolate(create_func_from_data(data.u))

        # trial and test functions
        self.mu_func = ufl.TrialFunction(self.scalar_space)
        self.v_func = ufl.TestFunction(self.vector_space)

    def solve(self, homogeneous=True):
        precomputed_derivatives = hasattr(self, 'Lu_func')

        # physical constants
        rho = 1000
        omega = 2 * np.pi * self.data.frequency.values.item()

        # construct bilinear form
        if precomputed_derivatives:
            Auv = -self.mu_func * ufl.inner(
                self.Lu_func, self.v_func
            ) * ufl.dx
        else:
            Auv = self.mu_func * ufl.inner(
                ufl.grad(self.u_func), ufl.grad(self.v_func)
            ) * ufl.dx

        if not homogeneous: # heterogeneous

            if precomputed_derivatives:
                Auv = Auv - ufl.inner(
                    ufl.grad(self.mu_func), self.Ju_func * self.v_func
                ) * ufl.dx
            else:
                Auv = Auv - ufl.inner(
                    ufl.grad(self.mu_func), ufl.grad(self.u_func) * self.v_func
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

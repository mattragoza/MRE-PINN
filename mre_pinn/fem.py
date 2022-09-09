import numpy as np
import xarray as xr
import scipy.ndimage
import scipy.interpolate

import ufl, dolfinx
from mpi4py import MPI

from .utils import as_xarray, minibatch
from . import discrete


class MultiFEM(object):
    '''
    An ensemble of FEMs for multifrequency inversion.
    '''
    def __init__(self, data, *args, **kwargs):
        self.data = data
        self.fems = [
            FEM(data.sel(frequency=[f]), *args, **kwargs) for f in data.frequency
        ]

    def solve(self, *args, **kwargs):
        frequencies = sorted(self.data.frequency)
        for freq, fem in zip(frequencies, self.fems):
            print(f'Solving FEM for frequency {freq}')
            fem.solve(*args, **kwargs)

    @minibatch
    def predict(self, x):

        # get indices that sort points
        # NOTE that argsort does not behave as expected here
        #   we want to sort first by frequency, then by space
        sort_indices = np.array(sorted(
            np.arange(len(x)), key=lambda i: tuple(x[i])
        ))
        x = x[sort_indices]

        # group by frequency and evaluate FEMs
        frequencies = sorted(self.data.frequency.values)

        results = None
        for freq, fem in zip(frequencies, self.fems):
            print(f'Predicting FEM for frequency {freq}')

            # subset of points at current frequency
            at_freq = np.isclose(x[:,0], freq)
            x_freq = x[at_freq][:,1:]
            if len(x_freq) == 0:
                continue

            # compute FEM results for current points
            freq_results = fem.predict(x_freq)

            if not results:
                results = list(freq_results)
                continue

            for i, y in enumerate(freq_results):
                results[i] = np.concatenate([results[i], y], axis=0)

        # return results in the original order
        unsort_indices = np.argsort(sort_indices)
        results = tuple(y[unsort_indices] for y in results)
        return results


class FEM(object):
    '''
    Solve Navier-Cauchy equation for steady-state
    elastic wave vibration using finite elements.
    '''
    def __init__(
        self,
        data,
        u_elem_type='CG-1',
        mu_elem_type='CG-1',
        align_nodes=False,
        savgol_filter=False,
    ):
        self.data = data
        x = data.field.spatial_points()

        # initialize the mesh
        mesh = create_mesh_from_data(data, align_nodes)

        # determine the FEM element types
        u_elem_type = parse_elem_type(u_elem_type)
        mu_elem_type = parse_elem_type(mu_elem_type)

        # define the FEM basis function spaces
        ndim = data.field.n_spatial_dims
        S = dolfinx.fem.FunctionSpace(mesh, mu_elem_type)
        V = dolfinx.fem.VectorFunctionSpace(mesh, u_elem_type, dim=ndim)
        T = dolfinx.fem.TensorFunctionSpace(mesh, u_elem_type, shape=(ndim, ndim))

        if savgol_filter: # Savitzky-Golay filtering

            # wave image-interpolating function
            Ku = discrete.savgol_smoothing(data.u, order=2, kernel_size=3)
            self.u_h = dolfinx.fem.Function(V)
            self.u_h.interpolate(create_func_from_data(Ku))

            # Jacobian-interpolating function
            Ju = discrete.savgol_jacobian(data.u, order=2, kernel_size=3)
            self.Ju_h = dolfinx.fem.Function(T)
            self.Ju_h.interpolate(create_func_from_data(Ju))

            # Laplacian-interpolating function
            Lu = discrete.savgol_laplacian(data.u, order=2, kernel_size=3)
            self.Lu_h = dolfinx.fem.Function(V)
            self.Lu_h.interpolate(create_func_from_data(Lu))
        else:
            # wave image-interpolating function
            self.u_h = dolfinx.fem.Function(V)
            self.u_h.interpolate(create_func_from_data(data.u))

        # trial and test functions
        self.mu_h = ufl.TrialFunction(S)
        self.v_h = ufl.TestFunction(V)

    def solve(self, homogeneous=True):
        from ufl import grad, inner, dx

        # physical constants
        rho = 1000
        omega = 2 * np.pi * self.data.frequency.values.item()

        # construct bilinear form
        if hasattr(self, 'Lu_h'): # precomputed derivatives
            Auv = -self.mu_h * inner(self.Lu_h, self.v_h) * dx
            if not homogeneous:
                Auv -= inner(grad(self.mu_h), self.Ju_h * self.v_h) * dx
        else:
            Auv = self.mu_h * inner(grad(self.u_h), grad(self.v_h)) * dx
            if not homogeneous:
                Auv -= inner(grad(self.mu_h), grad(self.u_h) * self.v_h) * dx

        # construct inner product
        bv = rho * omega**2 * inner(self.u_h, self.v_h) * dx

        # define the variational problem
        problem = dolfinx.fem.petsc.LinearProblem(
            Auv, bv, bcs=[], petsc_options={'ksp_type': 'lsqr', 'pc_type': 'none'}
        )
        self.mu_pred_h = problem.solve()

    def predict(self, x):
        mu_pred = eval_dolfinx_func(self.mu_pred_h, x)
        mu_pred = mu_pred.reshape(*self.data.mu.shape)
        mu_pred = as_xarray(mu_pred, like=self.data.mu)

        u_pred = eval_dolfinx_func(self.u_h, x)
        u_pred = u_pred.reshape(*self.data.u.shape)
        u_pred = as_xarray(u_pred, like=self.data.u)
        return u_pred, mu_pred


def parse_elem_type(s):
    '''
    Parse a string as an FEM element type
    using the format "{family}-{degree:d}",
    for instance: "CG-1", "CG-2", "DG-0"
    '''
    family, degree = s.split('-')
    return family, int(degree)


def grid_info_from_data(data):
    '''
    Args:
        data: An xarray with spatial dims.
    Returns:
        x_min, x_max, shape
    '''
    shape = np.array(data.field.spatial_shape)
    x = data.field.spatial_points()
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    return x_min, x_max, shape


def grid_to_mesh_info(x_min, x_max, shape, align_nodes):
    '''
    Args:
        x_min: The minimum grid point.
        x_max: The maximum grid point.
        shape: The number of grid points along
            each spatial dimension.
        align_nodes: If True, align mesh nodes to
            grid points, otherwise align centers
            of mesh cells to grid points.
    Returns:
        x_min: The minimum mesh node.
        x_max: The maximum mesh node.
        shape: The number of cells along
            each spatial dimension.
    '''
    # compute resolution in each dimension
    x_res = (x_max - x_min) / (shape - 1)

    if align_nodes: # align nodes to data
        shape = [d - 1 for d in shape]

    else: # align cells to data
        x_min -= x_res / 2
        x_max += x_res / 2
    
    return x_min, x_max, shape


def create_uniform_mesh(x_min, x_max, shape):
    '''
    Create a uniform hypercubic mesh with
    the provided spatial extent and shape.

    Args:
        x_min: The minimum mesh node.
        x_max: The maximum mesh node.
        shape: The number of cells along
            each spatial dimension.
    Returns:
        A dolfinx mesh.
    '''
    ndim = len(shape)

    # discretize the domain on a mesh
    if ndim == 3:
        raise NotImplementedError

    elif ndim == 2:
        mesh = dolfinx.mesh.create_rectangle(
            comm=MPI.COMM_WORLD,
            points=[x_min, x_max],
            n=shape,
            cell_type=dolfinx.mesh.CellType.triangle,
            diagonal=dolfinx.mesh.DiagonalType.left_right
        )
    elif ndim == 1:
        mesh = dolfinx.mesh.create_interval(
            comm=MPI.COMM_WORLD,
            points=[x_min, x_max],
            nx=shape[0]
        )
    
    return mesh


def mesh_info_from_data(data, align_nodes):
    x_min, x_max, shape = grid_info_from_data(data)
    return grid_to_mesh_info(x_min, x_max, shape, align_nodes)


def create_mesh_from_data(data, align_nodes):
    x_min, x_max, shape = mesh_info_from_data(data, align_nodes)
    return create_uniform_mesh(x_min, x_max, shape)


def create_func_from_data(data):
    '''
    Create a function that linearly linterpolates
    between the values of the provided xarray data,
    assuming they are defined on a spatial grid.

    Args:
        data: An xarray with D spatial dims.
    Returns:
        A function that linearly interpolates
            between the provided data values.
        Args:
            x_T: (3, N) array of spatial points.
        Returns:
            (K, N) array of interpolated values,
                where K is the product of non-spatial dims.
    '''
    ndim = data.field.n_spatial_dims
    x = data.field.spatial_points()
    y = data.field.values()

    if ndim > 1:
        interpolate = scipy.interpolate.LinearNDInterpolator(x, y)
    else:
        interpolate = scipy.interpolate.interp1d(
            x[:,0], y[:,0], bounds_error=False, fill_value='extrapolate'
        )

    def func(x_T):
        x = x_T[:ndim].T
        y = interpolate(x)
        y_T = y.reshape(len(x), -1).T
        return np.ascontiguousarray(y_T)

    return func


def get_containing_cells(mesh, x):
    '''
    Get indices of mesh cells that contain the given points.

    Args:
        mesh: A dolfinx mesh.
        x: (N, 3) array of spatial points.
    Returns:
        A list of indices of the cells containing
            each of the N spatial points.
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
        x: (N, D) array of spatial points.
    Returns:
        An (N, ...) array of f evaluated at x.
    '''
    x = np.concatenate([
        x, np.zeros((x.shape[0], 3 - x.shape[1]))
    ], axis=1)
    cells = get_containing_cells(f.function_space.mesh, x)
    return f.eval(x, cells)


def main(

    # data settings
    data_root='data/BIOQIC',
    data_name='fem_box',
    frequency=80,
    xyz_slice='2D',
    noise_ratio=0,

    # pde settings
    pde_name='hetero',

    # FEM settings
    u_elem_type='CG-1',
    mu_elem_type='CG-1',
    align_nodes=False,
    savgol_filter=False,

    # other settings
    save_prefix=None
):
    data, test_data = mre_pinn.data.load_bioqic_dataset(
        data_root=data_root,
        data_name=data_name,
        frequency=frequency,
        xyz_slice=xyz_slice,
        noise_ratio=noise_ratio
    )

    fem = FEM(
        data,
        u_elem_type=u_elem_type,
        mu_elem_type=mu_elem_type,
        align_nodes=align_nodes,
        savgol_filter=savgol_filter
    )

    # solve the variational problem
    fem.solve()

    # final test evaluation
    assert False, 'TODO'

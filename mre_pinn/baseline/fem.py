import numpy as np
import xarray as xr
import scipy.ndimage
import scipy.interpolate

import ufl, dolfinx
from mpi4py import MPI

from ..utils import print_if, as_xarray, minibatch, progress
from . import filters


def eval_fem_baseline(
    example,
    frequency,
    component=['x', 'y'],
    hetero=True,
    u_elem_type='CG-3',
    mu_elem_type='DG-1',
    align_nodes=True,
    mesh_scale=1,
    savgol_filter=False,
    order=2,
    kernel_size=3
):
    print('Evaluating FEM baseline')
    u = example.wave
    if not u.field.has_components:
        o = xr.zeros_like(u)
        new_dim = xr.DataArray(['x', 'y', 'z'], dims='component')
        u = xr.concat([u, o, o], dim=new_dim)

    mu = []
    for i, z in progress(list(enumerate(u.z))): # z slice
        if u.field.has_components:
            u_z = u.sel(z=z, component=component)
        else:
            u_z = u.sel(z=z)

        fem = MREFEM(
            u_z,
            u_elem_type=u_elem_type,
            mu_elem_type=mu_elem_type,
            align_nodes=align_nodes,
            mesh_scale=mesh_scale,
            savgol_filter=savgol_filter,
            order=order,
            kernel_size=kernel_size,
            verbose=False
        )
        fem.solve(frequency=frequency, hetero=hetero)

        # evaluate domain interior
        x_z = u_z.field.spatial_points(reshape=False)
        x_z[ 0,:,0] = x_z[ 1,:,0]
        x_z[-1,:,0] = x_z[-2,:,0]
        x_z[:, 0,1] = x_z[:, 1,1]
        x_z[:,-1,1] = x_z[:,-2,1]
        x_z = x_z.reshape(-1, x_z.shape[-1])
        mu_z = fem.predict(x_z)[1]
        mu_z = mu_z.reshape(u_z.field.spatial_shape)
        mu.append(mu_z)

    mu = np.stack(mu, axis=-1)
    if u.field.has_components:
        mu = as_xarray(mu, like=u.mean('component'))
    else:
        mu = as_xarray(mu, like=u)
    mu.name = 'baseline'
    example['fem'] = mu


class MREFEM(object):
    '''
    Solve Navier-Cauchy equation for steady-state
    elastic wave vibration using finite elements.
    '''
    def __init__(
        self,
        wave,
        u_elem_type='CG-3',
        mu_elem_type='DG-1',
        align_nodes=True,
        mesh_scale=1,
        savgol_filter=False,
        order=2,
        kernel_size=3,
        verbose=True
    ):
        # initialize the mesh
        print_if(verbose, 'Creating mesh from data')
        mesh = create_mesh_from_data(wave, align_nodes, mesh_scale)

        # determine the FEM element types
        u_elem_type = parse_elem_type(u_elem_type)
        mu_elem_type = parse_elem_type(mu_elem_type)

        # define the FEM basis function spaces
        print_if(verbose, 'Defining FEM basis function spaces')
        ndim = wave.field.n_spatial_dims
        S = dolfinx.fem.FunctionSpace(mesh, mu_elem_type)
        V = dolfinx.fem.VectorFunctionSpace(mesh, u_elem_type, dim=ndim)
        T = dolfinx.fem.TensorFunctionSpace(mesh, u_elem_type, shape=(ndim, ndim))

        print_if(verbose, 'Interpolating data into FEM basis')

        # wave image-interpolating function
        self.u_h = dolfinx.fem.Function(V if wave.field.has_components else S)
        self.u_h.interpolate(create_func_from_data(wave))

        # Jacobian-interpolating function
        Jwave = wave.field.gradient(
            savgol=savgol_filter, order=order, kernel_size=kernel_size, use_z=False
        )
        self.Ju_h = dolfinx.fem.Function(T if wave.field.has_components else V)
        self.Ju_h.interpolate(create_func_from_data(Jwave))

        # trial and test functions
        self.mu_h = ufl.TrialFunction(S)
        self.v_h = ufl.TestFunction(V if wave.field.has_components else S)

    def solve(self, rho=1000, frequency=None, hetero=False):
        from ufl import grad, div, transpose, inner, dx
        u_h, Ju_h, mu_h, v_h = self.u_h, self.Ju_h, self.mu_h, self.v_h

        omega = 2 * np.pi * frequency

        # construct bilinear form
        A = mu_h * inner(Ju_h, grad(v_h)) * dx
        if hetero:
            A -= inner(grad(mu_h), Ju_h * v_h) * dx

        # construct inner product
        b = rho * omega**2 * inner(u_h, v_h) * dx

        # define and solve the variational problem
        problem = dolfinx.fem.petsc.LinearProblem(
            A, b, bcs=[], petsc_options={'ksp_type': 'lsqr', 'pc_type': 'none'}
        )
        self.mu_pred_h = problem.solve()

    @minibatch
    def predict(self, x):
        u_pred  = eval_dolfinx_func(self.u_h, x)
        mu_pred = eval_dolfinx_func(self.mu_pred_h, x)
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


def grid_to_mesh_info(x_min, x_max, shape, align_nodes, mesh_scale):
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

    shape = [d // mesh_scale for d in shape]

    if align_nodes: # align nodes to data
        shape = [(d - 1) for d in shape]

    else: # align cells to data
        x_min += x_res / 2
        x_max -= x_res / 2

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
        mesh = dolfinx.mesh.create_box(
            comm=MPI.COMM_WORLD,
            points=[x_min, x_max],
            n=shape,
            cell_type=dolfinx.mesh.CellType.tetrahedron
        )

    elif ndim == 2:
        mesh = dolfinx.mesh.create_rectangle(
            comm=MPI.COMM_WORLD,
            points=[x_min, x_max],
            n=shape,
            cell_type=dolfinx.mesh.CellType.triangle
        )
    elif ndim == 1:
        mesh = dolfinx.mesh.create_interval(
            comm=MPI.COMM_WORLD,
            points=[x_min, x_max],
            nx=shape[0]
        )
    
    return mesh


def mesh_info_from_data(data, align_nodes, mesh_scale):
    x_min, x_max, shape = grid_info_from_data(data)
    return grid_to_mesh_info(x_min, x_max, shape, align_nodes, mesh_scale)


def create_mesh_from_data(data, align_nodes, mesh_scale):
    x_min, x_max, shape = mesh_info_from_data(data, align_nodes, mesh_scale)
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
    cell_indices = []
    for i, x_i in enumerate(x):
        try:
            cell_indices.append(cells.links(i)[0])
        except IndexError:
            mesh_min = mesh.geometry.x.min(axis=0)
            mesh_max = mesh.geometry.x.max(axis=0)
            msg = (
                f'Point {i} not contained in any mesh cell: {x_i}\n'
                f'Mesh bounds: ({mesh_min}, {mesh_max})\n'
            )
            raise ValueError(msg)
    return cell_indices


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

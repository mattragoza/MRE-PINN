import numpy as np
import xarray as xr
import torch
import deepxde

from ..utils import minibatch, as_xarray
from ..pde import laplacian
from .. import discrete


class PINNData(deepxde.data.Data):

    def __init__(self, arrays, pde, batch_size=None, device='cuda'):
        self.arrays = arrays
        self.pde = pde

        # convert to vector/scalar fields and coordinates
        #   while masking out the background region
        region = arrays.spatial_region.field.values()[:,0]

        self.x_train = arrays.field.points()[region >= 0]
        self.a_train = arrays.a.field.values()[region >= 0]
        self.u_train = arrays.u.field.values()[region >= 0]
        self.mu_train = arrays.mu.field.values()[region >= 0]
        self.Mu_train = arrays.Mu.field.values()[region >= 0]

        self.x_test = arrays.field.points()
        self.a_test = arrays.a.field.values()
        self.u_test = arrays.u.field.values()
        self.mu_test = arrays.mu.field.values()
        self.Mu_test = arrays.Mu.field.values()

        n_train = len(self.x_train)
        self.batch_sampler = deepxde.data.BatchSampler(n_train, shuffle=True)
        self.batch_size = batch_size
        self.device = device

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        (x, a), u_true = inputs, targets
        u_pred, mu_pred = outputs
        data_loss = loss_fn(u_true, u_pred)
        pde_res = self.pde(x, u_pred, mu_pred)
        pde_loss = loss_fn(0, pde_res)
        return [data_loss, pde_loss]

    def train_next_batch(self, batch_size=None):
        '''
        Args:
            batch_size
        Returns:
            inputs: A tuple of input arrays.
            targets: The target output array.
        '''
        batch_size = batch_size or self.batch_size
        inds = self.batch_sampler.get_next(batch_size)
        device = self.device
        x = torch.tensor(self.x_train[inds], device=device, dtype=torch.float32)
        a = torch.tensor(self.a_train[inds], device=device, dtype=torch.float32)
        u = torch.tensor(self.u_train[inds], device=device, dtype=torch.complex64)
        mu = torch.tensor(self.mu_train[inds], device=device, dtype=torch.complex64)
        Mu = torch.tensor(self.Mu_train[inds], device=device, dtype=torch.complex64)
        return (x, a), u, (mu, Mu)

    def test(self):
        x = torch.tensor(self.x_test, device=self.device, dtype=torch.float32)
        a = torch.tensor(self.a_test, device=self.device, dtype=torch.float32)
        u = torch.tensor(self.u_test, device=self.device, dtype=torch.complex64)
        mu = torch.tensor(self.mu_test, device=self.device, dtype=torch.complex64)
        Mu = torch.tensor(self.Mu_test, device=self.device, dtype=torch.complex64)
        return (x, a), u, (mu, Mu)


class PINNModel(deepxde.Model):

    def __init__(self, data, net, pde, batch_size=None):

        # initialize the training data
        data = PINNData(data, pde, batch_size)

        # initialize the network weights
        net.init_weights(
            inputs=(data.x_train, data.a_train),
            outputs=(data.u_train, data.mu_train)
        )
        super().__init__(data, net)

    @minibatch
    def predict(self, x, a):

        # compute model predictions
        x.requires_grad_(True)
        u_pred, mu_pred = self.net((x, a))

        # compute differential operators
        lu_pred = laplacian(u_pred, x, dim=1)
        f_trac, f_body = self.data.pde.traction_and_body_forces(x, u_pred, mu_pred)
        deepxde.gradients.clear()

        return u_pred, lu_pred, mu_pred, f_trac, f_body

    def test(self):
        
        # get ground truth and model predictions
        (x, a), u_true, (mu_true, Mu_base) = self.data.test()
        u_pred, lu_pred, mu_pred, f_trac, f_body = \
            self.predict(x, a, batch_size=self.batch_size)

        # convert tensors to xarrays
        u_true = self.data.arrays.u
        mu_true = self.data.arrays.mu
        Mu_base = self.data.arrays.Mu
        u_pred  = as_xarray(u_pred.reshape(u_true.shape), like=u_true)
        lu_pred = as_xarray(lu_pred.reshape(u_true.shape), like=u_true)
        f_trac  = as_xarray(f_trac.reshape(u_true.shape), like=u_true)
        f_body  = as_xarray(f_body.reshape(u_true.shape), like=u_true)
        mu_pred = as_xarray(mu_pred.reshape(mu_true.shape), like=mu_true)

        # compute discrete Laplacian of model wave field
        Lu_pred = discrete.laplacian(u_pred)

        # compute and concatenate model, residual, and reference values
        u_vars = ['u_pred', 'u_diff', 'u_true']
        u_dim = xr.DataArray(u_vars, dims=['variable'])
        u = xr.concat([u_pred, u_true - u_pred, u_true],  dim=u_dim)
        u.name = 'wave field'

        lu_vars = ['lu_pred', 'lu_diff', 'Lu_pred']
        lu_dim = xr.DataArray(lu_vars, dims=['variable'])
        lu = xr.concat([lu_pred, Lu_pred - lu_pred, Lu_pred], dim=lu_dim)
        lu.name = 'Laplacian'

        pde_vars = ['f_trac', 'f_sum', 'f_body']
        pde_dim = xr.DataArray(pde_vars, dims=['variable']) 
        pde = xr.concat([f_trac, f_trac + f_body, f_body], dim=pde_dim)
        pde.name = 'PDE'

        mu_vars = ['mu_pred', 'mu_diff', 'mu_true']
        mu_dim = xr.DataArray(mu_vars, dims=['variable'])
        mu = xr.concat([mu_pred, mu_true - mu_pred, mu_true], dim=mu_dim)
        mu.name = 'elastogram'

        Mu_vars = ['Mu', 'Mu_diff', 'mu_true']
        Mu_dim = xr.DataArray(Mu_vars, dims=['variable'])
        Mu = xr.concat([Mu_base, mu_true - Mu_base, mu_true], dim=Mu_dim)
        Mu.name = 'baseline'

        # take mean of mu across frequency
        mu = mu.mean('frequency')
        Mu = Mu.mean('frequency')

        return u, lu, pde, mu, Mu

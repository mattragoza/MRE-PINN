import time
from functools import cache
import numpy as np
import xarray as xr
import torch
import deepxde

from ..utils import concat, minibatch, as_xarray
from ..pde import laplacian
from .losses import msae_loss


class MREPINNData(deepxde.data.Data):

    def __init__(
        self,
        example,
        pde,
        loss_weights,
        pde_warmup_iters=10000,
        pde_init_weight=1e-19,
        pde_step_iters=5000,
        pde_step_factor=10,
        n_points=4096,
        device='cuda'
    ):
        self.example = example
        self.pde = pde

        metadata = example.metadata
        self.x_min = torch.tensor(metadata['origin'].wave, dtype=torch.float32)
        self.x_max = torch.tensor(metadata['limit'].wave, dtype=torch.float32)
        self.x_loc = torch.tensor(metadata['center'].wave, dtype=torch.float32)
        self.x_scale = torch.tensor(metadata['extent'].wave, dtype=torch.float32)

        stats = example.describe()
        self.u_loc = torch.tensor(stats['mean'].wave)
        self.u_scale = torch.tensor(stats['std'].wave)
        self.mu_loc = torch.tensor(stats['mean'].mre)
        self.mu_scale = torch.tensor(stats['std'].mre)

        if 'anat' in example:
            self.anatomical = True
            self.a_loc = torch.tensor(stats['mean'].anat)
            self.a_scale = torch.tensor(stats['std'].anat)
        else:
            self.anatomical = False
            self.a_loc = torch.zeros(0)
            self.a_scale = torch.zeros(0)

        if example.wave.field.has_components:
            self.wave_dims = example.wave.field.n_components
        else:
            self.wave_dims = 1

        self.loss_weights = loss_weights
        self.pde_warmup_iters = pde_warmup_iters
        self.pde_init_weight = pde_init_weight
        self.pde_step_iters = pde_step_iters
        self.pde_step_factor = pde_step_factor
        self.n_points = n_points
        self.device = device

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        x, = inputs
        u_true, mu_true, a_true = (
            targets[...,0:self.wave_dims],
            targets[...,self.wave_dims:self.wave_dims + 1],
            targets[...,self.wave_dims + 1:]
        )
        u_pred, mu_pred, a_pred = outputs

        u_loss  = loss_fn(u_true, u_pred)
        mu_loss = loss_fn(mu_true, mu_pred)
        a_loss  = loss_fn(a_true, a_pred) if self.anatomical else u_loss * 0

        pde_residual = self.pde(x, u_pred, mu_pred)
        pde_loss = loss_fn(0, pde_residual)

        u_weight, mu_weight, a_weight, pde_weight = self.loss_weights
        pde_iter = model.train_state.step - self.pde_warmup_iters
        if pde_iter < 0: # warmup phase (only train wave model)
            pde_weight = 0
        else: # PDE training phase
            n_steps = pde_iter // self.pde_step_iters
            pde_factor = self.pde_step_factor ** n_steps
            pde_weight = min(pde_weight, self.pde_init_weight * pde_factor)

        return [
            u_weight   * u_loss,
            mu_weight  * mu_loss,
            a_weight   * a_loss,
            pde_weight * pde_loss
        ]

    @cache
    def get_raw_tensors(self, device):

        # get numpy arrays from data example
        x = self.example.wave.field.points()
        u = self.example.wave.field.values()
        mu = self.example.mre.field.values()
        mu_mask = self.example.mre_mask.values.reshape(-1)

        # convert arrays to tensors on appropriate device
        x = torch.tensor(x, device=device, dtype=torch.float32)
        u = torch.tensor(u, device=device)
        mu = torch.tensor(mu, device=device)
        mu_mask = torch.tensor(mu_mask, device=device, dtype=torch.bool)

        if self.anatomical:
            a = self.example.anat.field.values()
            a = torch.tensor(a, device=device, dtype=torch.float32)
        else:
            a = u[:,:0]

        return x, u, mu, mu_mask, a

    def get_tensors(self, use_mask=True, normalize=True):
        x, u, mu, mu_mask, a = self.get_raw_tensors(self.device)

        if use_mask: # apply mask and subsample points
            x, u, mu = x[mu_mask], u[mu_mask], mu[mu_mask]
            sample = torch.randperm(x.shape[0])[:self.n_points]
            x, u, mu = x[sample], u[sample], mu[sample]
            a = a[mu_mask][sample]

        if normalize: # center and scale
            x = (x - self.x_loc) / self.x_scale
            u = (u - self.u_loc) / self.u_scale
            mu = (mu - self.mu_loc) / self.mu_scale
            a = (a - self.a_loc) / self.a_scale

        input_ = (x,)
        target = torch.cat([u, mu, a], dim=-1)
        aux_var = ()
        return input_, target, aux_var

    def train_next_batch(self, batch_size=None, **kwargs):
        '''
        Returns:
            inputs: Tuple of input tensors.
            targets: Target tensor.
            aux_vars: Tuple of auxiliary tensors.
        '''
        return self.get_tensors(**kwargs)

    def test(self, **kwargs):
        return self.get_tensors(**kwargs)


class MREPINNModel(deepxde.Model):

    def __init__(self, example, net, pde, **kwargs):

        # initialize the training data
        data = MREPINNData(example, pde, **kwargs)

        super().__init__(data, net)

    def benchmark(self, n_iters=100):

        print(f'# iterations: {n_iters}')
        data_time = 0
        model_time = 0
        loss_time = 0
        for i in range(n_iters):
            t_start = time.time()
            inputs, targets, aux_vars = self.data.train_next_batch()
            t_data = time.time()
            x, = inputs
            x.requires_grad = True
            outputs = self.net(inputs)
            t_model = time.time()
            losses = self.data.losses(targets, outputs, msae_loss, inputs, self)
            t_loss = time.time()
            data_time += (t_data - t_start) / n_iters
            model_time += (t_model - t_data) / n_iters
            loss_time += (t_loss - t_model) / n_iters

        iter_time = data_time + model_time + loss_time
        print(f'Data time/iter:  {data_time:.4f}s ({data_time/iter_time*100:.2f}%)')
        print(f'Model time/iter: {model_time:.4f}s ({model_time/iter_time*100:.2f}%)')
        print(f'Loss time/iter:  {loss_time:.4f}s ({loss_time/iter_time*100:.2f}%)')
        print(F'Total time/iter: {iter_time:.4f}s')

        total_time = iter_time * n_iters
        print(f'Total time: {total_time:.4f}s')
        print(f'1k iters time: {iter_time * 1e3 / 60:.2f}m')
        print(f'10k iters time: {iter_time * 1e4 / 60:.2f}m')
        print(f'100k iters time: {iter_time * 1e5 / 3600:.2f}h')

    def predict(self, x, batch_size=None):

        def predict_batch(x):
            x.requires_grad = True
            u_pred, mu_pred, a_pred = self.net.forward(inputs=(x,))
            u_pred = u_pred * self.data.u_scale + self.data.u_loc
            mu_pred = mu_pred * self.data.mu_scale + self.data.mu_loc
            a_pred = a_pred * self.data.a_scale + self.data.a_loc
            lu_pred = laplacian(u_pred, x)
            f_trac, f_body = self.data.pde.traction_and_body_forces(
                x, u_pred, mu_pred
            )
            return u_pred, mu_pred, a_pred, lu_pred, f_trac, f_body

        if batch_size is None:
            return predict_batch(x)

        outputs = []
        for i in range(0, x.shape[0], batch_size):
            batch_outputs = predict_batch(x[i:i + batch_size])
            outputs.append(batch_outputs)

        return map(concat, zip(*outputs))

    def test(self, test_vars='ulpm'):
        
        # get input tensors
        inputs, targets, aux_vars = self.data.test(use_mask=False, normalize=True)

        # get model predictions as tensors
        u_pred, mu_pred, a_pred, lu_pred, f_trac, f_body = \
            self.predict(*inputs, batch_size=self.data.n_points)

        # get ground truth xarrays
        u_true = self.data.example.wave
        mu_true = self.data.example.mre
        if 'anat' in self.data.example:
            a_true = self.data.example.anat
        else:
            a_true = u_true * 0
            a_pred = u_pred * 0
        mu_mask = self.data.example.mre_mask
        Lu_true = self.data.example.Lu

        u_shape = u_true.shape
        mu_shape = mu_true.shape
        a_shape = a_true.shape

        # adjust mask level
        mask_level = 1.0
        mu_mask = ((mu_mask > 0) - 1) * mask_level + 1

        mu_pred = as_xarray(mu_pred.reshape(mu_shape), like=mu_true)
        u_pred  = as_xarray(u_pred.reshape(u_shape), like=u_true)
        f_trac  = as_xarray(f_trac.reshape(u_shape), like=u_true)
        f_body  = as_xarray(f_body.reshape(u_shape), like=u_true)

        return_vars = []
        if 'a' in test_vars:
            a_pred  = as_xarray(a_pred.reshape(a_shape), like=a_true)
            a_vars = ['a_pred', 'a_diff', 'a_true']
            a_dim = xr.DataArray(a_vars, dims=['variable'])
            a = xr.concat([
                mu_mask * a_pred,
                mu_mask * (a_true - a_pred),
                mu_mask * a_true
            ], dim=a_dim)
            a.name = 'anatomy'
            return_vars.append(a)

        if 'u' in test_vars:
            u_vars = ['u_pred', 'u_diff', 'u_true']
            u_dim = xr.DataArray(u_vars, dims=['variable'])
            u = xr.concat([
                mu_mask * u_pred,
                mu_mask * (u_true - u_pred),
                mu_mask * u_true
            ], dim=u_dim)
            u.name = 'wave field'
            return_vars.append(u)

        if 'l' in test_vars:
            lu_pred = as_xarray(lu_pred.reshape(u_shape), like=u_true)
            lu_vars = ['lu_pred', 'lu_diff', 'Lu_true']
            lu_dim = xr.DataArray(lu_vars, dims=['variable'])
            lu = xr.concat([
                mu_mask * lu_pred,
                mu_mask * (Lu_true - lu_pred),
                mu_mask * Lu_true
            ], dim=lu_dim)
            lu.name = 'Laplacian'
            return_vars.append(lu)

        if 'p' in test_vars:
            pde_vars = ['pde_grad', 'pde_diff', 'mu_diff']
            pde_dim = xr.DataArray(pde_vars, dims=['variable'])
            pde_grad = -((f_trac + f_body) * lu_pred * 2)
            if 'component' in pde_grad.sizes:
                pde_grad = pde_grad.sum('component')
            pde_grad *= self.data.loss_weights[2]
            mu_diff = mu_true - mu_pred
            pde = xr.concat([
                mu_mask * pde_grad,
                mu_mask * (mu_diff - pde_grad),
                mu_mask * mu_diff
            ], dim=pde_dim)
            pde.name = 'PDE'
            return_vars.append(pde)

        if 'm' in test_vars:
            mu_vars = ['mu_pred', 'mu_diff', 'mu_true']
            mu_dim = xr.DataArray(mu_vars, dims=['variable'])
            mu = xr.concat([
                mu_mask * mu_pred,
                mu_mask * mu_diff,
                mu_mask * mu_true
            ],dim=mu_dim)
            mu.name = 'elastogram'
            return_vars.append(mu)

        if 'd' in test_vars:
            mu_direct = self.data.example.direct
            direct_vars = ['direct_pred', 'direct_diff', 'mu_true']
            direct_dim = xr.DataArray(direct_vars, dims=['variable'])
            direct = xr.concat([
                mu_mask * mu_direct,
                mu_mask * (mu_true - mu_direct),
                mu_mask * mu_true
            ], dim=direct_dim)
            direct.name = 'direct'
            return_vars.append(direct)

        if 'f' in test_vars:
            mu_fem = self.data.example.fem
            fem_vars = ['fem_pred', 'fem_diff', 'mu_true']
            fem_dim = xr.DataArray(fem_vars, dims=['variable'])
            fem = xr.concat([
                mu_mask * mu_fem,
                mu_mask * (mu_true - mu_fem),
                mu_mask * mu_true
            ], dim=fem_dim)
            fem.name = 'FEM'
            return_vars.append(f)

        return 'train', return_vars

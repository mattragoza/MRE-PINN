import time
from functools import cache
import numpy as np
import xarray as xr
import torch
import deepxde

from ..utils import minibatch, as_xarray
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
        patch_size=1,
        device='cuda'
    ):
        self.example = example
        self.pde = pde

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

        self.anatomical = ('anat' in example)
        self.patch_size = patch_size

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
        a_loss  = loss_fn(a_true, a_pred)

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
    def get_raw_tensors(self, device, patch_size):
        example = self.example

        # get numpy arrays from data example
        x = example.wave.field.points()
        u = example.wave.field.values()
        mu = example.mre.field.values()
        mu_mask = example.mre_mask.values.reshape(-1)

        # convert arrays to tensors on appropriate device
        x = torch.tensor(x, device=device, dtype=torch.float32)
        u = torch.tensor(u, device=device)
        mu = torch.tensor(mu, device=device)
        mu_mask = torch.tensor(mu_mask, device=device, dtype=torch.bool)

        if self.anatomical: # compute anatomical image patches
            a = example.anat.values.transpose(2, 3, 0, 1) # xyzc → zcxy
            a = torch.tensor(a, device=device, dtype=torch.float32)
            k = patch_size
            a = torch.nn.functional.unfold(a, kernel_size=k, padding=k // 2)
            a = torch.permute(a, (2, 0, 1)) # z(ck)(xy) → (xy)z(ck)
            a = a.reshape(-1, a.shape[-1])
        else:
            a = x[:,:0]
        return x, u, mu, mu_mask, a

    def get_tensors(self, use_mask=True):
        x, u, mu, mu_mask, a = self.get_raw_tensors(self.device, self.patch_size)

        if use_mask: # apply mask and subsample points
            x, u, mu = x[mu_mask], u[mu_mask], mu[mu_mask]
            sample = torch.randperm(x.shape[0])[:self.n_points]
            x, u, mu = x[sample], u[sample], mu[sample]
            if self.anatomical:
                a = a[mu_mask][sample]

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

    @minibatch
    def predict(self, x):
        x.requires_grad = True
        u_pred, mu_pred, a_pred = self.net.forward(inputs=(x,))
        lu_pred = laplacian(u_pred, x)
        f_trac, f_body = self.data.pde.traction_and_body_forces(x, u_pred, mu_pred)
        return (
            u_pred.detach().cpu(),
            mu_pred.detach().cpu(),
            a_pred.detach().cpu(),
            lu_pred.detach().cpu(),
            f_trac.detach().cpu(),
            f_body.detach().cpu()
       )

    def test(self):
        
        # get input tensors
        inputs, targets, aux_vars = self.data.test(use_mask=False)

        # get model predictions as tensors
        u_pred, mu_pred, a_pred, lu_pred, f_trac, f_body = \
            self.predict(*inputs, batch_size=self.data.n_points)

        # get ground truth xarrays
        u_true = self.data.example.wave
        mu_true = self.data.example.mre
        a_true = self.data.example.anat
        mu_direct = self.data.example.direct
        mu_fem = self.data.example.fem
        mu_mask = self.data.example.mre_mask
        Lu_true = self.data.example.Lu

        # apply mask level
        mask_level = 1.0
        mu_mask = ((mu_mask > 0) - 1) * mask_level + 1

        # convert predicted tensors to xarrays
        u_shape, mu_shape, a_shape = u_true.shape, mu_true.shape, a_true.shape
        u_pred  = as_xarray(u_pred.reshape(u_shape), like=u_true)
        lu_pred = as_xarray(lu_pred.reshape(u_shape), like=u_true)
        f_trac  = as_xarray(f_trac.reshape(u_shape), like=u_true)
        f_body  = as_xarray(f_body.reshape(u_shape), like=u_true)
        mu_pred = as_xarray(mu_pred.reshape(mu_shape), like=mu_true)
        a_pred  = as_xarray(a_pred.reshape(a_shape), like=a_true)

        a_vars = ['a_pred', 'a_diff', 'a_true']
        a_dim = xr.DataArray(a_vars, dims=['variable'])
        a = xr.concat([
            mu_mask * a_pred,
            mu_mask * (a_true - a_pred),
            mu_mask * a_true
        ], dim=a_dim)
        a.name = 'anatomy'

        u_vars = ['u_pred', 'u_diff', 'u_true']
        u_dim = xr.DataArray(u_vars, dims=['variable'])
        u = xr.concat([
            mu_mask * u_pred,
            mu_mask * (u_true - u_pred),
            mu_mask * u_true
        ], dim=u_dim)
        u.name = 'wave field'

        lu_vars = ['lu_pred', 'lu_diff', 'Lu_true']
        lu_dim = xr.DataArray(lu_vars, dims=['variable'])
        lu = xr.concat([
            mu_mask * lu_pred,
            mu_mask * (Lu_true - lu_pred),
            mu_mask * Lu_true
        ], dim=lu_dim)
        lu.name = 'Laplacian'

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

        mu_vars = ['mu_pred', 'mu_diff', 'mu_true']
        mu_dim = xr.DataArray(mu_vars, dims=['variable'])
        mu = xr.concat([
            mu_mask * mu_pred,
            mu_mask * mu_diff,
            mu_mask * mu_true
        ],dim=mu_dim)
        mu.name = 'elastogram'

        direct_vars = ['direct_pred', 'direct_diff', 'mu_true']
        direct_dim = xr.DataArray(direct_vars, dims=['variable'])
        direct = xr.concat([
            mu_mask * mu_direct,
            mu_mask * (mu_true - mu_direct),
            mu_mask * mu_true
        ], dim=direct_dim)
        direct.name = 'direct'

        fem_vars = ['fem_pred', 'fem_diff', 'mu_true']
        fem_dim = xr.DataArray(fem_vars, dims=['variable'])
        fem = xr.concat([
            mu_mask * mu_fem,
            mu_mask * (mu_true - mu_fem),
            mu_mask * mu_true
        ], dim=fem_dim)
        fem.name = 'FEM'

        return 'train', (a, u, lu, pde, mu, direct, fem)

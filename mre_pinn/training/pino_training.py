import time
from functools import cache
import numpy as np
import xarray as xr
import torch
import torchvision.transforms.functional as TF
import deepxde

from ..utils import as_xarray
from ..pde import laplacian
from .. import discrete
from .losses import msae_loss


def affine_transform(a, rotate, translate, scale):
    '''
    Args:
        a: (n_x, n_y, n_z, n_channels) input tensor
        rotate, translate, scale: Transform parameters
    Returns:
        t: (n_x, n_y, n_z, n_channels) transformerd tensor
    '''
    a = torch.permute(a, (3, 2, 1, 0)) # xyzc->czyx
    mode = TF.InterpolationMode.BILINEAR
    b = TF.affine(
        a, rotate, list(translate), scale, shear=0, interpolation=mode, fill=0
    )
    b = torch.permute(a, (3, 2, 1, 0)) # czyx->xyzc
    return b


class PINOData(deepxde.data.Data):

    def __init__(
        self,
        train_set,
        test_set,
        pde,
        loss_weights,
        pde_warmup_iters=10000,
        pde_step_iters=5000,
        pde_step_factor=10,
        n_points=4096,
        batch_size=None,
        device='cuda'
    ):
        self.train_set = train_set
        self.test_set = test_set
        self.pde = pde

        self.loss_weights = loss_weights
        self.pde_warmup_iters = pde_warmup_iters
        self.pde_step_iters = pde_step_iters
        self.pde_step_factor = pde_step_factor
        self.pde_init_weight = 1e-19
        self.n_points = n_points

        self.train_sampler = deepxde.data.BatchSampler(len(train_set), shuffle=True)
        self.test_sampler = deepxde.data.BatchSampler(len(test_set), shuffle=False)
        self.batch_size = batch_size
        self.device = device

        print('Precomputing tensors')
        for idx in range(len(train_set)):
            self.get_raw_tensors(train_set, idx)
        for idx in range(len(test_set)):
            self.get_raw_tensors(test_set, idx)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        a_im, u_im, x = inputs
        u_true, mu_true = torch.split(targets, 1, dim=-1)
        u_pred, mu_pred = outputs[:2]
        u_loss = loss_fn(u_true, u_pred)
        mu_true_mean = mu_true.mean(dim=(1,2), keepdim=True)
        mu_pred_mean = mu_pred.mean(dim=(1,2), keepdim=True)
        mu_loss = loss_fn(mu_true_mean, mu_pred_mean)
        pde_residual = self.pde(x, u_pred, mu_pred)
        pde_loss = loss_fn(0, pde_residual)

        u_weight, mu_weight, pde_weight = self.loss_weights
        pde_iter = model.train_state.step - self.pde_warmup_iters
        if pde_iter < 0: # warmup phase (only train wave model)
            pde_weight = 0
        else: # PDE training phase
            n_steps = pde_iter // self.pde_step_iters
            pde_factor = self.pde_step_factor ** n_steps
            pde_weight = min(pde_weight, self.pde_init_weight * pde_factor)
        return [
            u_loss * u_weight, mu_loss * mu_weight, pde_loss * pde_weight
        ]

    @cache
    def get_raw_tensors(self, dataset, idx):
        '''
        Args:
            dataset: Train or test dataset.
            idx: Example index in dataset.
        Returns:
            a, u, x, mu, mask: Raw tensors
        '''
        example_id = dataset.example_ids[idx]
        example = dataset.examples[example_id]

        # get numpy arrays from data example
        #   ensure that they have a channel dim
        a = example.anat.transpose('x', 'y', 'z', 'sequence').values
        u = example.wave.values[...,None]
        x = example.wave.field.points(reshape=False) * 1e-3
        mu = example.mre.values[...,None]
        mask = example.mre_mask.values

        # convert arrays to tensors on appropriate device
        a = torch.tensor(a, device=self.device, dtype=torch.float32)
        u = torch.tensor(u, device=self.device, dtype=torch.float32)
        x = torch.tensor(x, device=self.device, dtype=torch.float32)
        mu = torch.tensor(mu, device=self.device, dtype=torch.float32)
        mask = torch.tensor(u, device=self.device, dtype=torch.bool)

        return a, u, x, mu, mask

    def get_tensors(self, dataset, idx, augment, use_mask):
        
        a_im, u_im, x_im, mu_im, mask = self.get_raw_tensors(dataset, idx)

        if augment: # apply data augmentation
            rotate = np.random.uniform(-8, 8, 1)[0]
            translate = np.random.uniform(-10, 10, 2)
            scale = np.random.uniform(0.9, 1.1, 1)[0]
            a_im = affine_transform(a_im, rotate, translate, scale)
            u_im = affine_transform(u_im, rotate, translate, scale)
            x_im = affine_transform(x_im, -rotate, -translate, 1 / scale)

        # reshape field points and values
        x = x_im.view(-1, 3)
        u = u_im.view(-1, 1)
        mu = mu_im.view(-1, 1)
        mask = mask.view(-1)

        if use_mask: # apply mask and subsample points
            x, u, mu = x[mask], u[mask], mu[mask]
            sample = torch.randperm(x.shape[0])[:self.n_points]
            x, u, mu = x[sample], u[sample], mu[sample]

        input_ = (a_im, u_im, x)
        target = torch.cat([u, mu], dim=-1)
        aux_var = ()
        return input_, target, aux_var

    def get_next_batch(
        self,
        dataset,
        batch_sampler,
        batch_size=None,
        augment=True,
        use_mask=True,
        return_inds=False
    ):
        '''
        Args:
            dataset: Train or test dataset.
            batch_size: Number of examples in batch.
        Returns:
            inputs: Tuple of input tensors.
            targets: Tuple of target tensors.
            aux_vars: Tuple of auxiliary tensors.
        '''
        t_start = time.time()

        batch_size = batch_size or self.batch_size
        batch_inds = batch_sampler.get_next(batch_size)

        inputs, targets, aux_vars = [], [], []
        for idx in batch_inds:
            input, target, aux = self.get_tensors(dataset, idx, augment, use_mask)

            inputs.append(input)
            targets.append(target)
            aux_vars.append(aux)

        inputs = tuple(torch.stack(x) for x in zip(*inputs))
        targets = torch.stack(targets)
        aux_vars = tuple(torch.stack(x) for x in zip(*aux_vars))

        if return_inds:
            return inputs, targets, aux_vars, batch_inds
        else:
            return inputs, targets, aux_vars

    def train_next_batch(self, batch_size=None):
        return self.get_next_batch(self.train_set, self.train_sampler, batch_size)

    def test(self, batch_size=1, **kwargs):
        return self.get_next_batch(
            self.test_set, self.test_sampler, batch_size, **kwargs
        )


class PINOModel(deepxde.Model):
    
    def __init__(self, train_set, test_set, net, pde, **kwargs):

        # initialize the training data
        data = PINOData(train_set, test_set, pde, **kwargs)

        # initialize the network weights
        #TODO net.init_weights()

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
            a, u, x = inputs
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

    def predict(self, inputs):
        a, u, x = inputs
        x.requires_grad = True
        u_pred, mu_pred = self.net.forward(inputs, debug=True)
        lu_pred = laplacian(u_pred, x)
        f_trac, f_body = self.data.pde.traction_and_body_forces(x, u_pred, mu_pred)
        return u_pred, mu_pred, lu_pred, f_trac, f_body

    def test(self):
        
        # get model predictions as tensors
        inputs, targets, aux_vars, inds = self.data.test(
            augment=False, use_mask=False, return_inds=True
        )
        u_pred, mu_pred, lu_pred, f_trac, f_body = self.predict(inputs)
        #Mu_pred = -1000 * (2 * np.pi * 80)**2 * u_pred / lu_pred

        # get ground truth xarrays
        a_true  = self.data.test_set[inds[0]].anat
        u_true  = self.data.test_set[inds[0]].wave
        mu_true = self.data.test_set[inds[0]].mre
        a_mask  = self.data.test_set[inds[0]].anat_mask
        m_mask  = self.data.test_set[inds[0]].mre_mask
        Lu_true = self.data.test_set[inds[0]].Lwave
        Mu_base = self.data.test_set[inds[0]].Mwave

        # apply mask level
        mask_level = 1.0
        a_mask = (a_mask - 1) * mask_level + 1
        m_mask = (m_mask - 1) * mask_level + 1

        # convert predicted tensors to xarrays
        u_shape, mu_shape = u_true.shape, mu_true.shape
        u_pred  = as_xarray(u_pred.reshape(u_shape), like=u_true)
        lu_pred = as_xarray(lu_pred.reshape(u_shape), like=u_true)
        f_trac  = as_xarray(f_trac.reshape(u_shape), like=u_true)
        f_body  = as_xarray(f_body.reshape(u_shape), like=u_true)
        #Mu_pred = as_xarray(Mu_pred.reshape(mu_shape), like=Mu_true)
        mu_pred = as_xarray(mu_pred.reshape(mu_shape), like=mu_true)
        
        # combine xarrays into single xarray
        if False:
            a_vars = ['a_mask', 'a_over', 'a_true']
            a_dim = xr.DataArray(a_vars, dims=['variable'])
            a = xr.concat([a_mask, a_mask * a_true, a_true], dim=a_dim)
            a.name = 'anatomy'

        u_vars = ['u_pred', 'u_diff', 'u_true']
        u_dim = xr.DataArray(u_vars, dims=['variable'])
        u = xr.concat(
            [u_pred * m_mask, (u_true - u_pred) * m_mask, u_true * m_mask],
            dim=u_dim
        )
        u.name = 'wave field'

        lu_vars = ['lu_pred', 'lu_diff', 'Lu_true']
        lu_dim = xr.DataArray(lu_vars, dims=['variable'])
        lu = xr.concat(
            [lu_pred * m_mask, (Lu_true - lu_pred) * m_mask, Lu_true * m_mask],
            dim=lu_dim
        )
        lu.name = 'Laplacian'

        pde_vars = ['f_trac', 'pde_diff', 'pde_grad']
        pde_dim = xr.DataArray(pde_vars, dims=['variable'])
        pde = xr.concat(
            [f_trac * m_mask, (f_trac + f_body) * m_mask, 2 * lu_pred * (f_trac + f_body) * m_mask],
            dim=pde_dim
        )
        pde.name = 'PDE'

        mu_vars = ['mu_pred', 'mu_diff', 'mu_true']
        mu_dim = xr.DataArray(mu_vars, dims=['variable'])
        mu = xr.concat(
            [mu_pred * m_mask, (mu_true - mu_pred) * m_mask, mu_true * m_mask],
            dim=mu_dim
        )
        mu.name = 'elastogram'

        Mu_vars = ['Mu_base', 'Mu_diff', 'mu_true']
        Mu_dim = xr.DataArray(Mu_vars, dims=['variable'])
        Mu = xr.concat(
            [Mu_base * m_mask, (Mu_base - mu_true) * m_mask, mu_true * m_mask],
            dim=Mu_dim
        )
        Mu.name = 'baseline'

        return u, lu, pde, mu, Mu

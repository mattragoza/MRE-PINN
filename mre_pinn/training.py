import numpy as np
import xarray as xr
import torch
import deepxde

from .utils import as_xarray, as_matrix, minibatch
from . import pde, discrete, visual


class MREPINNModel(deepxde.Model):

    def __init__(self, net, pde, geom, bc, batch_size=None, **kwargs):
        bc.set_batch_size(batch_size)
        data = deepxde.data.PDE(geom, pde, bc, **kwargs)
        super().__init__(data, net)

    @minibatch
    def predict(self, x):

        # compute model predictions
        x = torch.as_tensor(x)
        x.requires_grad_(True)
        outputs = self.net(x)
        u_pred, mu_pred = outputs[:,:-1], outputs[:,-1:]

        # compute differential operators
        lu_pred = pde.laplacian(u_pred, x, dim=1)
        f_trac, f_body = self.data.pde.traction_and_body_forces(x, outputs)
        deepxde.gradients.clear()

        return u_pred, lu_pred, mu_pred, f_trac, f_body


class PeriodicCallback(deepxde.callbacks.Callback):

    def __init__(self, period):
        super().__init__()
        self.period = period

    def on_batch_begin(self):
        if self.model.train_state.step % self.period != 0:
            return
        self.on_period_begin()

    def on_batch_end(self):
        if (self.model.train_state.step + 1) % self.period != 0:
            return
        self.on_period_end()

    def on_period_begin(self):
        pass

    def on_period_end(self):
        pass


class PDEResampler(PeriodicCallback):
    '''
    Resample PDE and BC training points each period.
    '''
    def on_period_end(self):
        self.model.data.train_x_all = None
        self.model.data.train_x_bc = None
        self.model.data.resample_train_points()


class TestEvaluation(PeriodicCallback):

    def __init__(self, period, data, batch_size, **kwargs):
        super().__init__(period)
        self.data = data
        self.batch_size = batch_size
        self.viewer_kws = kwargs
        self.initialized = False

    def on_period_begin(self):

        # get ground truth values
        u, mu, Lu = self.data.u, self.data.mu, self.data.Lu
        x = u.field.points().astype(np.float32)

        u_pred, lu_pred, mu_pred, f_trac, f_body = \
            self.model.predict(x, batch_size=self.batch_size)

        # convert tensors to xarrays
        u_pred  = as_xarray(u_pred.reshape(u.shape),   like=u)
        lu_pred = as_xarray(lu_pred.reshape(u.shape),  like=u)
        mu_pred = as_xarray(mu_pred.reshape(mu.shape), like=mu)
        f_trac  = as_xarray(f_trac.reshape(u.shape),   like=u)
        f_body  = as_xarray(f_body.reshape(u.shape),   like=u)

        # compute discrete Laplacian
        Lu = discrete.laplacian(u_pred)


        # compute and concatenate residuals wrt reference values
        new_dim = xr.DataArray(['u_pred', 'residual', 'u_true'], dims=['residual'])
        u  = xr.concat([u_pred,  u  - u_pred,  u],  dim=new_dim)

        new_dim = xr.DataArray(['lu_pred', 'residual', 'Lu_pred'], dims=['residual'])
        lu = xr.concat([lu_pred, Lu - lu_pred, Lu], dim=new_dim)

        new_dim = xr.DataArray(['f_trac', 'residual', 'f_body'], dims=['residual'])
        pde = xr.concat([f_trac, f_trac + f_body, f_body], dim=new_dim)

        new_dim = xr.DataArray(['mu_pred', 'residual', 'mu_true'], dims=['residual'])
        mu = xr.concat([mu_pred, mu - mu_pred, mu], dim=new_dim)

        # take mean of mu across frequency
        mu = mu.mean('frequency')

        if not self.initialized:
            pct = 95
            u_map = visual.wave_color_map()
            u_max = np.percentile(np.abs(u), pct) * 1.1
            self.u_kws = dict(cmap=u_map, vmax=u_max)
            self.u_kws.update(self.viewer_kws)
            self.u_viewer = visual.XArrayViewer(u, **self.u_kws)

            lu_max = np.percentile(np.abs(lu), pct) * 1.1
            self.lu_kws = dict(cmap=u_map, vmax=lu_max)
            self.lu_kws.update(self.viewer_kws)
            self.lu_viewer = visual.XArrayViewer(lu, **self.lu_kws)

            pde_max = np.percentile(np.abs(pde), pct) * 1.1
            self.pde_kws = dict(cmap=u_map, vmax=pde_max)
            self.pde_kws.update(self.viewer_kws)
            self.pde_viewer = visual.XArrayViewer(pde, **self.pde_kws)

            mu_map = visual.elast_color_map(symmetric=True)
            mu_max = np.percentile(np.abs(mu), pct) * 6
            self.mu_kws = dict(cmap=mu_map, vmax=mu_max)
            self.mu_kws.update(self.viewer_kws)
            self.mu_viewer = visual.XArrayViewer(mu, **self.mu_kws)
            self.initialized = True
        else:
            self.u_viewer.update_array(u)
            self.lu_viewer.update_array(lu)
            self.pde_viewer.update_array(pde)
            self.mu_viewer.update_array(mu)


class SummaryDisplay(deepxde.display.TrainingDisplay):
    '''
    Training display that only prints a summary at
    the end of training instead of after every test.
    '''
    def print_one(self, *args, **kwargs):
        return


def normalized_l2_loss_fn(y):
    norm = np.linalg.norm(y).mean()
    def loss_fn(y_true, y_pred):
        return torch.mean(
            torch.norm(y_true - y_pred, dim=1) / norm
        )
    return loss_fn

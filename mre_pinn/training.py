import time
import numpy as np
import pandas as pd
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

    def __init__(
        self,
        data,
        batch_size,
        test_every=1000,
        save_every=10000,
        save_prefix=None,
        plot=True,
        view=True,
        interact=False
    ):
        assert save_every % test_every == 0
        super().__init__(period=test_every)
        self.data = data.copy()
        self.batch_size = batch_size
        self.plot = plot
        self.view = view
        self.interact = interact

        index_cols = [
            'iteration',
            'variable_type',
            'variable_source',
            'variable_name',
            'spatial_frequency_bin',
            'spatial_region',
        ]
        self.metrics = pd.DataFrame(columns=index_cols)
        self.metrics.set_index(index_cols, inplace=True)
        self.save_prefix = save_prefix
        self.save_every = save_every

        # estimate % of time spent testing
        self.t_start = time.time()
        self.test_time = 0

    def on_period_begin(self):
        t_start = time.time()
        save = (self.model.train_state.step % self.save_every == 0)
        self.test_evaluate(self.data, save)
        self.test_time += (time.time() - t_start)

    def on_period_end(self):
        total_time = time.time() - self.t_start
        pct_test_time = self.test_time / total_time * 100
        print(f'Time spent testing: {pct_test_time:.2f}%')

    def test_evaluate(self, data, save=True):    
        arrays = self.compute_arrays(self.data)
        metrics = self.compute_metrics(arrays)
        self.update_arrays(arrays)
        self.update_metrics(metrics)
        if self.plot:
            self.update_plots()
        if self.view:
            self.update_viewers()
        if save and self.save_prefix: # save model state
            self.model.save(self.save_prefix + '_model')

    def compute_arrays(self, data):

        # get ground truth values
        u_true = data.u
        mu_true = data.mu
        Mu_base = data.Mu
        x = data.field.points().astype(np.float32)

        # get model predictions
        u_pred, lu_pred, mu_pred, f_trac, f_body = \
            self.model.predict(x, batch_size=self.batch_size)

        # convert tensors to xarrays
        u_pred  = as_xarray(u_pred.reshape(u_true.shape), like=u_true)
        lu_pred = as_xarray(lu_pred.reshape(u_true.shape), like=u_true)
        f_trac  = as_xarray(f_trac.reshape(u_true.shape), like=u_true)
        f_body  = as_xarray(f_body.reshape(u_true.shape), like=u_true)
        mu_pred = as_xarray(mu_pred.reshape(mu_true.shape), like=mu_true)

        # compute discrete Laplacian of model wave field
        Lu_pred = discrete.laplacian(u_pred)

        # compute and concatenate model, residual, and reference values
        new_dim = xr.DataArray(
            ['u_pred', 'u_diff', 'u_true'], dims=['variable']
        )
        u_diff = u_true - u_pred
        u = xr.concat([u_pred, u_diff, u_true],  dim=new_dim)
        u.name = 'wave field'

        new_dim = xr.DataArray(
            ['lu_pred', 'lu_diff', 'Lu_pred'], dims=['variable']
        )
        lu_diff = Lu_pred - lu_pred
        lu = xr.concat([lu_pred, lu_diff, Lu_pred], dim=new_dim)
        lu.name = 'Laplacian'

        new_dim = xr.DataArray(
            ['f_trac', 'f_sum', 'f_body'], dims=['variable']
        )
        f_sum = f_trac + f_body
        pde = xr.concat([f_trac, f_sum, f_body], dim=new_dim)
        pde.name = 'PDE'

        new_dim = xr.DataArray(
            ['mu_pred', 'mu_diff', 'mu_true'], dims=['variable']
        )
        mu_diff = mu_true - mu_pred
        mu = xr.concat([mu_pred, mu_diff, mu_true], dim=new_dim)
        mu.name = 'elastogram'

        new_dim = xr.DataArray(
            ['Mu', 'Mu_diff', 'mu_true'], dims=['variable']
        )
        Mu_diff = mu_true - Mu_base
        Mu = xr.concat([Mu_base, Mu_diff, mu_true], dim=new_dim)
        Mu.name = 'baseline'

        # take mean of mu across frequency
        mu = mu.mean('frequency')
        Mu = Mu.mean('frequency')

        return u, lu, pde, mu, Mu

    def compute_metrics(self, arrays):
        
        # current training iteration
        iter_ = self.model.train_state.step
        sources = ['model', 'residual', 'reference']

        metrics = []
        for array in arrays:
            var_type = array.name

            for var_src, var_name in zip(sources, array['variable'].values):
                a = array.sel(variable=var_name)
                value = np.mean(np.abs(a)**2)
                index = (iter_, var_type, var_src, var_name, 'all', 'all')
                metric = (index, 'mean_squared_abs_value', value)
                metrics.append(metric)

                ps = discrete.power_spectrum(a)
                for f_bin, value in zip(ps.spatial_frequency_bins.values, ps.values):
                    index = (iter_, var_type, var_src, var_name, f_bin.right, 'all')
                    metric = (index, 'power_density', value)
                    metrics.append(metric)

                mav = np.abs(a).groupby('spatial_region').median(...)
                for region, value in zip(mav.spatial_region.values, mav.values):
                    index = (iter_, var_type, var_src, var_name, 'all', region)
                    metric = (index, 'mean_abs_value', value)
                    metrics.append(metric)

        return metrics

    def update_arrays(self, arrays, save=True):
        self.arrays = arrays
        if self.save_prefix and save:
            for array in arrays:
                array_name = array.name.lower().replace(' ', '_')
                array = xr.concat([array.real, array.imag], dim='part')
                array.to_netcdf(f'{self.save_prefix}_{array_name}.nc')

    def update_metrics(self, new_metrics, save=True):
        for index, name, value in new_metrics:
            self.metrics.loc[index, name] = value
        if self.save_prefix and save:
            self.metrics.to_csv(self.save_prefix + '_train_metrics.csv', sep=' ')

    def update_plots(self, save=True):
        data = self.metrics.reset_index()
        try:
            self.norm_plot.update_data(data)
            self.freq_plot.update_data(data[data.variable_source == 'residual'])
            self.region_plot.update_data(data[data.variable_type == 'elastogram'])
        except AttributeError:
            self.norm_plot = visual.DataViewer(
                data,
                x='iteration',
                y='mean_squared_abs_value',
                col='variable_type',
                hue='variable_source'
            )
            self.freq_plot = visual.DataViewer(
                data[data.variable_source == 'residual'],
                x='iteration',
                y='power_density',
                col='variable_type',
                #row='variable_source',
                hue='spatial_frequency_bin',
                palette='Blues_r',
            )
            self.region_plot = visual.DataViewer(
                data[data.variable_type == 'elastogram'],
                x='iteration',
                y='mean_abs_value',
                col='variable_source',
                #row='variable_type',
                hue='spatial_region',
                palette='Greens_r'
            )
        if self.save_prefix and save:
            self.norm_plot.to_png(self.save_prefix + '_train_norms.png') 
            self.freq_plot.to_png(self.save_prefix + '_train_freqs.png')
            self.region_plot.to_png(self.save_prefix + '_train_regions.png')

    def update_viewers(self, save=True):
        arrays = self.arrays
        try: # update array values
            for i, array in enumerate(arrays):
                self.viewers[i].update_array(array)
        except AttributeError: # initialize viewers
            self.viewers = []
            for i, array in enumerate(arrays):
                kwargs = visual.get_color_kws(array)
                viewer = visual.XArrayViewer(
                    array,
                    row='domain',
                    col='variable',
                    y='y' if 'y' in self.data.field.spatial_dims else None,
                    ax_width=2,
                    interact=self.interact,
                    **kwargs
                )
                self.viewers.append(viewer)
        if self.save_prefix and save:
            curr_iter = self.model.train_state.step
            for array, viewer in zip(arrays, self.viewers):
                array_name = array.name.lower().replace(' ', '_')
                viewer.to_png(f'{self.save_prefix}_{array_name}_{curr_iter}.png')


class SummaryDisplay(deepxde.display.TrainingDisplay):
    '''
    Training display that only prints a summary at
    the end of training instead of after every test.
    '''
    def print_one(self, *args, **kwargs):
        return


def normalized_l2_loss_fn(y):
    norm = np.linalg.norm(y, axis=1).mean()
    def loss_fn(y_true, y_pred):
        return torch.mean(
            torch.norm(y_true - y_pred, dim=1) / norm
        )
    return loss_fn


def standardized_msae_loss_fn(y):
    variance = torch.var(torch.as_tensor(y))
    def loss_fn(y_true, y_pred):
        return torch.mean(
            torch.abs(y_true - y_pred)**2 / variance
        )
    return loss_fn

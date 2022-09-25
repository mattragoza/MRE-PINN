import os, time
import numpy as np
import xarray as xr
import pandas as pd
import deepxde

from .utils import as_xarray
from .training import PeriodicCallback
from . import discrete, visual


class TestEvaluator(PeriodicCallback):

    def __init__(
        self,
        data,
        model,
        batch_size=None,
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
        self.model = model
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

        if save_prefix: # create output subdirectories
            save_dir, save_name = os.path.split(save_prefix)
            viewer_dir = os.path.join(save_dir, 'viewers')
            weight_dir = os.path.join(save_dir, 'weights')
            os.makedirs(viewer_dir, exist_ok=True)
            os.makedirs(weight_dir, exist_ok=True)
            self.viewer_prefix = os.path.join(viewer_dir, save_name)
            self.weight_prefix = os.path.join(weight_dir, save_name)

        # estimate % of time spent testing
        self.t_start = time.time()
        self.test_time = 0

    @property
    def iteration(self):
        try:
            return self.model.train_state.step
        except AttributeError:
            return 0

    def on_period_begin(self):
        t_start = time.time()
        save_model = (self.iteration % self.save_every == 0)
        self.test(self.data, save_model)
        self.test_time += (time.time() - t_start)

    def on_period_end(self):
        total_time = time.time() - self.t_start
        pct_test_time = self.test_time / total_time * 100
        print(f'Time spent testing: {pct_test_time:.2f}%')

    def test(self, data, save_model=True):    
        arrays = self.compute_arrays(self.data)
        metrics = self.compute_metrics(arrays)
        self.update_arrays(arrays)
        self.update_metrics(metrics)
        if self.plot:
            self.update_plots()
        if self.view:
            self.update_viewers()
        if save_model and self.weight_prefix: # save model state
            self.model.save(self.weight_prefix + '_model')

    def compute_arrays(self, data):

        # get ground truth values
        u_true = data.u
        mu_true = data.mu
        Mu_base = data.Mu
        x = data.field.points()

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
        iter_ = self.iteration
        sources = ['model', 'residual', 'reference']

        metrics = []
        for array in arrays: # wave field, laplacian, elastogram, baseline
            var_type = array.name

            # model, residual, or reference
            for var_src, var_name in zip(sources, array['variable'].values):
                a = array.sel(variable=var_name)
                value = np.mean(np.abs(a)**2)
                index = (iter_, var_type, var_src, var_name, 'all', 'all')
                metric = (index, 'mean_squared_abs_value', value)
                metrics.append(metric)

                psd = discrete.power_spectrum(a)
                for f_bin, value in zip(psd.spatial_frequency_bins.values, psd.values):
                    index = (iter_, var_type, var_src, var_name, f_bin.right, 'all')
                    metric = (index, 'power_density', value)
                    metrics.append(metric)

                mav = np.abs(a.real).groupby('spatial_region').median(...)
                for region, value in zip(mav.spatial_region.values, mav.values):
                    index = (iter_, var_type, var_src, var_name, 'all', region)
                    metric = (index, 'median_abs_value', value)
                    metrics.append(metric)

        return metrics

    def update_arrays(self, arrays, save=True):
        self.arrays = arrays
        if save and self.save_prefix:
            for array in arrays:
                array_name = array.name.lower().replace(' ', '_')
                array = xr.concat([array.real, array.imag], dim='part')
                array.to_netcdf(f'{self.save_prefix}_{array_name}.nc')

    def update_metrics(self, new_metrics, save=True):
        for index, name, value in new_metrics:
            self.metrics.loc[index, name] = value
        if save and self.save_prefix:
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
                y='median_abs_value',
                col='variable_source',
                #row='variable_type',
                hue='spatial_region',
                palette='Greens_r'
            )
        if save and self.save_prefix:
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
        if save and self.viewer_prefix:
            curr_iter = self.iteration
            for array, viewer in zip(arrays, self.viewers):
                array_name = array.name.lower().replace(' ', '_')
                viewer.to_png(f'{self.viewer_prefix}_{array_name}_{curr_iter}.png')

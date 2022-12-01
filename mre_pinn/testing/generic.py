import os, time
import numpy as np
import pandas as pd
import xarray as xr

from ..training.callbacks import PeriodicCallback
from .. import visual


class TestEvaluator(PeriodicCallback):

    def __init__(
        self,
        test_every=1000,
        save_every=10000,
        save_prefix=None,
        plot=True,
        view=True,
        interact=False
    ):
        assert save_every % test_every == 0
        super().__init__(period=test_every)
        self.plot = plot
        self.view = view
        self.interact = interact

        index_cols = [
            'iteration',
            'dataset',
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
        self.test(save_model)

        total_time = time.time() - self.t_start
        test_time = time.time() - t_start
        self.test_time += test_time
        pct_test_time = self.test_time / total_time * 100
        print(f'Time spent testing: {test_time:.4f} ({pct_test_time:.2f}%)')

    def test(self, save_model=True):    
        dataset, arrays = self.model.test()
        self.update_arrays(arrays)
        metrics = self.compute_metrics(dataset, arrays)
        self.update_metrics(metrics)
        if self.plot:
            self.update_plots()
        if self.view:
            self.update_viewers()
        if save_model and self.save_prefix: # save model state
            self.model.save(self.weight_prefix + '_model')

    def compute_metrics(self, dataset, arrays):
        
        # current training iteration
        iter_ = self.iteration
        sources = ['model', 'residual', 'reference']

        metrics = []
        for array in arrays: # wave field, Laplacian, PDE, elastogram, direct, FEM
            var_type = array.name

            # model, residual, or reference
            for var_src, var_name in zip(sources, array['variable'].values):
                a = array.sel(variable=var_name)
                value = np.mean(np.abs(a)**2)
                index = (iter_, dataset, var_type, var_src, var_name, 'all', 'all')
                metric = (index, 'MSAV', value)
                metrics.append(metric)

                psd = power_spectral_density(a)
                for f_bin, value in zip(psd.spatial_frequency_bins.values, psd.values):
                    index = (
                        iter_, dataset, var_type, var_src, var_name, f_bin.right, 'all'
                    )
                    metric = (index, 'PSD', value)
                    metrics.append(metric)

                mav = np.abs(a).groupby('region').median(...)
                for region, value in zip(mav.region.values, mav.values):
                    index = (
                        iter_, dataset, var_type, var_src, var_name, 'all', region
                    )
                    metric = (index, 'MAV', value)
                    metrics.append(metric)

            # also compute correlation
            pred_var, diff_var, true_var = array['variable'].values
            a_pred = array.sel(variable=pred_var)
            a_true = array.sel(variable=true_var)
            corr = xr.corr(np.abs(a_pred), np.abs(a_true))
            index = (iter_, dataset, var_type, var_src, var_name, 'all', 'all')
            metric = (index, 'R', value)
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
            self.corr_plot.update_data(data)
            self.freq_plot.update_data(data)
            self.region_plot.update_data(data)
        except AttributeError:
            self.norm_plot = visual.DataViewer(
                data,
                x='iteration',
                y='MSAV',
                col='variable_type',
                row='variable_source',
                hue='dataset',
                ax_height=1.5,
                ax_width=1.25
            )
            self.corr_plot = visual.DataViewer(
                data,
                x='iteration',
                y='R',
                col='variable_type',
                hue='dataset',
                ax_height=1.5,
                ax_width=1.25
            )
            self.freq_plot = visual.DataViewer(
                data,
                x='iteration',
                y='PSD',
                col='variable_type',
                row='variable_source',
                hue='spatial_frequency_bin',
                ax_height=1.5,
                ax_width=1.25
            )
            self.region_plot = visual.DataViewer(
                data,
                x='iteration',
                y='MAV',
                col='variable_type',
                row='variable_source',
                hue='spatial_region',
                ax_height=1.5,
                ax_width=1.25
            )
        if save and self.save_prefix:
            self.norm_plot.to_png(self.save_prefix + '_train_norms.png') 
            self.corr_plot.to_png(self.save_prefix + '_train_corrs.png')
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
                var = array['variable'][2]
                kwargs = visual.get_color_kws(array.sel(variable=var))
                viewer = visual.XArrayViewer(
                    array,
                    row='part' if array.field.is_complex else None,
                    col='variable',
                    y='y' if 'y' in array.coords else None,
                    ax_height=2,
                    ax_width=2,
                    interact=self.interact,
                    polar=array.name in {'elastogram', 'baseline', 'FEM', 'direct'},
                    **kwargs
                )
                self.viewers.append(viewer)
        if save and self.save_prefix:
            curr_iter = self.iteration
            for array, viewer in zip(arrays, self.viewers):
                array_name = array.name.lower().replace(' ', '_')
                viewer.to_png(f'{self.viewer_prefix}_{array_name}_{curr_iter}.png')


def power_spectral_density(u, n_bins=10):
    '''
    Compute power density wrt spatial frequency.
    '''
    # compute power spectrum
    ps = np.abs(u.field.fft())**2
    ps.name = u.name

    # compute spatial frequency radii for binning
    x = ps.field.spatial_points(reshape=False, standardize=True)
    r = np.linalg.norm(x, ord=2, axis=-1)
    ps = ps.assign_coords(spatial_frequency=(ps.field.spatial_dims, r * n_bins))

    # take mean across spatial frequency bins
    bins = np.linspace(0, n_bins, n_bins + 1, endpoint=True)
    ps = ps.groupby_bins('spatial_frequency', bins=bins).mean(...)
    return ps #.values

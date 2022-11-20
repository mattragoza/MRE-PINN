import os, pathlib, glob
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.model_selection import KFold

from ..utils import exists, print_if, as_xarray, is_iterable
from ..visual import XArrayViewer

import scipy.ndimage
from .. import discrete


class MREDataset(object):
    '''
    A set of preprocessed MRE imaging sequences in xarray format.
    '''
    def __init__(self, example_ids, examples):
        self.example_ids = np.array(example_ids)
        self.examples = examples

    @classmethod
    def from_bioqic(cls, bioqic):
        examples = {}
        example_ids = []
        for frequency in bioqic.arrays.frequency:
            ex = MREExample.from_bioqic(bioqic, frequency)
            example_ids.append(ex.example_id)
            examples[ex.example_id] = ex
        return MREDataset(example_ids, examples)

    @classmethod
    def from_cohort(cls, cohort):
        examples = {}
        example_ids = []
        for pid in cohort.patient_ids:
            ex = MREExample.from_patient(cohort.patients[pid])
            example_ids.append(ex.example_id)
            examples[ex.example_id] = ex
        return MREDataset(example_ids, examples)

    @classmethod
    def load_xarrays(cls, xarray_dir, anat=False, verbose=True):
        examples = {}
        example_ids = []
        for sub_dir in sorted(os.listdir(xarray_dir)):
            ex = MREExample.load_xarrays(xarray_dir, sub_dir, anat)
            example_ids.append(ex.example_id)
            examples[ex.example_id] = ex
        return MREDataset(example_ids, examples)

    def save_xarrays(self, xarray_dir, verbose=True):
        for xid in self.example_ids:
            self.examples[xid].save_xarrays(xarray_dir, verbose)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if is_iterable(idx, string_ok=False) or isinstance(idx, slice):
            example_ids = self.example_ids[idx]
            examples = {xid:self.examples[xid] for xid in example_ids}
            return MREDataset(example_ids, examples)
        else:
            return self.examples[self.example_ids[idx]]

    @property
    def metadata(self):
        dfs = []
        for xid in self.example_ids:
            df = self.examples[xid].metadata
            df['example_id'] = xid
            dfs.append(df)
        df = pd.concat(dfs).reset_index()
        return df.set_index(['example_id', 'variable', 'dimension'])

    def describe(self):
        dfs = []
        for xid in self.example_ids:
            df = self.examples[xid].describe()
            df['example_id'] = xid
            dfs.append(df)
        df = pd.concat(dfs).reset_index()
        return df.set_index(['example_id', 'variable', 'component'])

    def eval_baseline(self, **kwargs):
        for xid in self.example_ids:
            self.examples[xid].eval_baseline(**kwargs)

    def shuffle(self, seed=None):
        np.random.seed(seed)
        np.random.shuffle(self.example_ids)

    def k_fold_split(self, *args, **kwargs):
        k_fold = KFold(*args, **kwargs)
        for train_ids, test_ids in k_fold.split(self.example_ids):
            yield self[train_ids], self[test_ids]


class MREExample(object):
    '''
    A single instance of preprocessed MRE imaging sequences.
    '''
    def __init__(self, example_id, wave, mre, mre_mask, **arrays):
        self.example_id = example_id
        self.arrays = {'wave': wave, 'mre': mre, 'mre_mask': mre_mask}
        self.arrays.update({k: v for k, v in arrays.items() if exists(v)})

    @classmethod
    def from_bioqic(cls, bioqic, frequency):
        example_id = str(frequency.item())
        arrays = bioqic.arrays.sel(frequency=frequency)
        example = MREExample(
            example_id,
            wave=arrays['wave'],
            mre=arrays['mu'],
            mre_mask=arrays['spatial_region']
        )
        if bioqic.anat_var is not None:
            example['anat'] = arrays['anat']
            example['anat_mask'] = arrays['spatial_region']
        return example

    @classmethod
    def from_patient(cls, patient):
        example_id = patient.patient_id
        arrays = patient.convert_images()
        sequences = ['t1_pre_in', 't1_pre_out', 't1_pre_water', 't1_pre_fat', 't2']
        new_dim = xr.DataArray(sequences, dims=['sequence'])
        return MREExample(
            example_id,
            wave=arrays['wave'],
            mre=arrays['mre'],
            mre_mask=arrays['mre_mask'],
            anat=xr.concat([arrays[a] for a in sequences], dim=new_dim),
            anat_mask=arrays['anat_mask']
        )

    @classmethod
    def load_xarrays(cls, xarray_dir, example_id, anat=False, verbose=True):
        xarray_dir  = pathlib.Path(xarray_dir)
        example_dir = xarray_dir / str(example_id)
        wave = load_xarray_file(example_dir / 'wave.nc', verbose)
        mre  = load_xarray_file(example_dir / 'mre.nc',  verbose)
        mre_mask  = load_xarray_file(example_dir / 'mre_mask.nc',  verbose)
        example = MREExample(example_id, wave, mre, mre_mask)
        if anat:
            anat = load_xarray_file(example_dir / 'anat.nc', verbose)
            anat_mask = load_xarray_file(example_dir / 'anat_mask.nc', verbose)
            example['anat'] = anat
            example['anat_mask'] = anat_mask
        return example

    def save_xarrays(self, xarray_dir, verbose=True):
        xarray_dir  = pathlib.Path(xarray_dir)
        example_dir = xarray_dir / str(self.example_id)
        example_dir.mkdir(parents=True, exist_ok=True)
        save_xarray_file(example_dir / 'wave.nc', self.wave, verbose)
        save_xarray_file(example_dir / 'mre.nc',  self.mre,  verbose)
        save_xarray_file(example_dir / 'mre_mask.nc',  self.mre_mask,  verbose)
        if 'anat' in self:
            save_xarray_file(example_dir / 'anat.nc', self.anat, verbose)
            save_xarray_file(example_dir / 'anat_mask.nc', self.anat_mask, verbose)

    def __getitem__(self, key):
        return self.arrays[key]

    def __setitem__(self, key, val):
        self.arrays[key] = val

    def __contains__(self, key):
        return key in self.arrays

    def __getattr__(self, key):
        if key in self.arrays:
            return self.arrays[key]
        raise AttributeError(f"'MREExample' object has no attribute '{key}'")

    def vars(self):
        return self.arrays.keys()

    @property
    def metadata(self):
        index_cols = ['variable', 'dimension']
        df = pd.DataFrame(columns=index_cols).set_index(index_cols)
        for var, array in self.arrays.items():
            shape = array.field.spatial_shape
            res = array.field.spatial_resolution
            origin = array.field.origin
            for i, dim in enumerate(array.field.spatial_dims):
                df.loc[(var, dim), 'size'] = shape[i]
                df.loc[(var, dim), 'spacing'] = res[i]
                df.loc[(var, dim), 'origin'] = origin[i]
        df['size'] = df['size'].astype(int)
        df['limit'] = df['origin'] + (df['size'] - 1) * df['spacing']
        df['center'] = df['origin'] + (df['size'] - 1) / 2 * df['spacing']
        df['extent'] = df['limit'] - df['origin'] + df['spacing']
        return df

    def describe(self):
        index_cols = ['variable', 'component']
        df = pd.DataFrame(columns=index_cols).set_index(index_cols)
        for var, array in self.arrays.items():
            if not array.field.has_components:
                comp = 'scalar'
                values = array.values
                df.loc[(var, comp), 'dtype'] = values.dtype
                df.loc[(var, comp), 'count'] = values.size
                df.loc[(var, comp), 'mean'] = values.mean()
                df.loc[(var, comp), 'std'] = values.std()
                df.loc[(var, comp), 'min'] = values.min()
                df.loc[(var, comp), '25%'] = np.percentile(values, 25)
                df.loc[(var, comp), '50%'] = np.percentile(values, 50)
                df.loc[(var, comp), '75%'] = np.percentile(values, 75)
                df.loc[(var, comp), 'max'] = values.max()
                continue
            for comp in array.component.values:
                values = array.sel(component=comp).values
                df.loc[(var, comp), 'dtype'] = values.dtype
                df.loc[(var, comp), 'count'] = values.size
                df.loc[(var, comp), 'mean'] = values.mean()
                df.loc[(var, comp), 'std'] = values.std()
                df.loc[(var, comp), 'min'] = values.min()
                df.loc[(var, comp), '25%'] = np.percentile(values, 25)
                df.loc[(var, comp), '50%'] = np.percentile(values, 50)
                df.loc[(var, comp), '75%'] = np.percentile(values, 75)
                df.loc[(var, comp), 'max'] = values.max()
        df['count'] - df['count'].astype(int)
        return df

    def eval_baseline(
        self, order=3, kernel_size=5, rho=1e3, frequency=60, polar=True,
        postprocess=False
    ):
        # Savitsky-Golay smoothing and derivatives
        u = self.wave
        Ku = u.copy() # smoothed wave field
        Lu = u.copy() # smoothed Laplacian
        for z in range(u.shape[2]): # z slice
            if not u.field.has_components:
                Ku[...,z] = discrete.savgol_smoothing(
                    u[...,z], order=order, kernel_size=kernel_size
                )
                Lu[...,z] = discrete.savgol_laplacian(
                    u[...,z], order=order, kernel_size=kernel_size
                )
                continue
            for c in range(u.shape[3]): # component
                Ku[...,z,c] = discrete.savgol_smoothing(
                    u[...,z,c], order=order, kernel_size=kernel_size
                )
                Lu[...,z,c] = discrete.savgol_laplacian(
                    u[...,z,c], order=order, kernel_size=kernel_size
                )

        # algebraic Helmholtz inversion
        Mu = discrete.helmholtz_inversion(Ku, Lu, rho, frequency, polar, eps=1e-5)
        Mu.name = 'baseline'

        # post-processing
        if postprocess:
            Mu.values[Mu.values < 0] = 0
            k = np.array([1, 2, 3, 2, 1])
            k = np.einsum('i,j,k->ijk', k, k, k)
            Mu_median = scipy.ndimage.median_filter(Mu, footprint=k > 2)
            Mu_outliers = np.abs(Mu - Mu_median) > 1000
            Mu.values = np.where(Mu_outliers, Mu_median, Mu)
            Mu.values = scipy.ndimage.gaussian_filter(Mu, sigma=0.65, truncate=3)

        # store results
        self.arrays['base'] = Mu

    def view(self, *args, **kwargs):
        for var in (args or self.arrays):
            XArrayViewer(self.arrays[var], **kwargs)


def save_xarray_file(nc_file, array, verbose=True):
    print_if(verbose, f'Writing {nc_file}')
    if np.iscomplexobj(array):
        new_dim = xr.DataArray(['real', 'imag'], dims=['part'])
        array = xr.concat([array.real, array.imag], dim=new_dim)
    array.to_netcdf(nc_file)


def load_xarray_file(nc_file, verbose=True):
    print_if(verbose, f'Loading {nc_file}')
    array = xr.open_dataarray(nc_file)
    if 'part' in array.dims:
        real = array.sel(part='real')
        imag = array.sel(part='imag')
        return real + 1j * imag
    else:
        return array

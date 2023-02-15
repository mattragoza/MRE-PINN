import os, pathlib, glob
from functools import cache
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.model_selection import KFold

from ..utils import exists, print_if, as_xarray, is_iterable, progress
from ..visual import XArrayViewer


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
            examples = {xid: self.examples[xid] for xid in example_ids}
            return MREDataset(example_ids, examples)
        else:
            return self.examples[self.example_ids[idx]]

    @property
    @cache
    def metadata(self):
        dfs = []
        for xid in self.example_ids:
            df = self.examples[xid].metadata
            df['example_id'] = xid
            dfs.append(df)
        df = pd.concat(dfs).reset_index()
        return df.set_index(['example_id', 'variable', 'dimension'])

    @cache
    def describe(self):
        dfs = []
        for xid in progress(self.example_ids):
            df = self.examples[xid].describe()
            df['example_id'] = xid
            dfs.append(df)
        df = pd.concat(dfs).reset_index()
        return df.set_index(['example_id', 'variable', 'component'])

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
    def __init__(self, example_id, wave, mre, mre_mask, anat=None):
        self.example_id = example_id
        wave = wave.assign_coords(region=mre_mask)
        mre = mre.assign_coords(region=mre_mask)
        self.arrays = {'wave': wave, 'mre': mre, 'mre_mask': mre_mask}
        if anat is not None:
            self.arrays['anat'] = anat.assign_coords(region=mre_mask)

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
        new_dim = xr.DataArray(sequences, dims=['component'])
        anat = xr.concat([arrays[a] for a in sequences], dim=new_dim)
        anat = anat.transpose('x', 'y', 'z', 'component')
        example = MREExample(
            example_id,
            wave=arrays['wave'],
            mre=arrays['mre'],
            mre_mask=arrays['mre_mask'],
            anat=anat
        )
        example['anat_mask'] = arrays['anat_mask']
        return example

    @classmethod
    def load_xarrays(cls, xarray_dir, example_id, anat=False, verbose=True):
        xarray_dir  = pathlib.Path(xarray_dir)
        example_dir = xarray_dir / str(example_id)
        wave = load_xarray_file(example_dir / 'wave.nc', verbose)
        mre  = load_xarray_file(example_dir / 'mre.nc',  verbose)
        mre_mask  = load_xarray_file(example_dir / 'mre_mask.nc',  verbose)
        if anat:
            anat = load_xarray_file(example_dir / 'anat.nc', verbose)
            anat_mask = load_xarray_file(example_dir / 'anat_mask.nc', verbose)
            if 'sequence' in anat.coords:
                anat = anat.rename(sequence='component')
            example = MREExample(example_id, wave, mre, mre_mask, anat=anat)
        else:
            example = MREExample(example_id, wave, mre, mre_mask)
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

    def downsample(self, **factors):
        arrays = {}
        for var_name in self.vars():
            array = self[var_name].coarsen(boundary='trim', **factors)
            if var_name in {'mre_mask', 'anat_mask', 'spatial_region'}:
                array = array.max()
            else:
                array = array.mean()
            arrays[var_name] = array
        return MREExample(self.example_id, **arrays)

    def add_gaussian_noise(self, noise_ratio, axis=None):
        self.arrays['wave'] = add_gaussian_noise(self.wave, noise_ratio, axis)

    def view(self, *args, mask=0, **kwargs):
        for var_name in (args or self.arrays):
            array = self.arrays[var_name]
            if mask > 0:
                if var_name == 'anat':
                    m = self.arrays['mre_mask']
                else:
                    m = self.arrays['mre_mask']
                m = ((m > 0) - 1) * mask + 1
                array = as_xarray(array * m, like=array)
            XArrayViewer(array, **kwargs)


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
        if 'part' not in array.coords:
            array['part'] = xr.DataArray(['real', 'imag'], dims='part')
        real = array.sel(part='real')
        imag = array.sel(part='imag')
        return real + 1j * imag
    else:
        return array


def complex_normal(loc, scale, size):
    radius = np.random.randn(*size) * scale 
    angle = np.random.rand(*size) * 2 * np.pi
    return radius * np.exp(1j * angle) + loc


def add_gaussian_noise(array, noise_ratio, axis=None):
    array_abs = np.abs(array)
    array_mean = np.mean(array_abs, axis=axis, keepdims=True)
    array_variance = np.var(array_abs, axis=axis, keepdims=True)
    array_power = array_mean**2 + array_variance
    noise_power = noise_ratio * array_power
    noise_std = np.sqrt(noise_power).values
    if np.iscomplexobj(array):
        noise = complex_normal(loc=0, scale=noise_std, size=array.shape)
    else:
        noise = np.random.normal(loc=0, scale=noise_std, size=array.shape)
    return array + noise

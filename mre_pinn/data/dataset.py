import os, pathlib, glob
import numpy as np
import xarray as xr
from sklearn.model_selection import KFold

from ..utils import print_if, as_xarray, is_iterable
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
        example_ids = {}
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

    def eval_baseline(self):
        for xid in self.example_ids:
            ex = self.examples[xid]
            ex.eval_baseline()

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
    def __init__(self, example_id, anat, wave, mre, mre_mask, anat_mask):
        self.example_id = example_id
        self.anat = anat
        self.wave = wave
        self.mre = mre
        self.mre_mask = mre_mask
        self.anat_mask = anat_mask

    @classmethod
    def from_bioqic(cls, bioqic, frequency):
        example_id = str(frequency.item())
        arrays = bioqic.arrays.sel(frequency=frequency)
        anat = arrays['anat'] if bioqic.anat_var else None
        wave = arrays['wave']
        mre  = arrays['mu']
        mre_mask = arrays['spatial_region']
        anat_mask = arrays['spatial_region'] if bioqic.anat_var else None
        return cls(example_id, anat, wave, mre, mre_mask, anat_mask)

    @classmethod
    def from_patient(cls, patient):
        example_id = patient.patient_id
        xarrays = patient.convert_images()
        anat_seqs = ['t1_pre_in', 't1_pre_out', 't1_pre_water', 't1_pre_fat', 't2']
        anat_seq_dim = xr.DataArray(anat_seqs, dims=['sequence'])
        anat = xr.concat([xarrays[a] for a in anat_seqs], dim=anat_seq_dim)
        wave = xarrays['wave']
        mre = xarrays['mre']
        mre_mask = xarrays['mre_mask']
        anat_mask = xarrays['anat_mask']
        return cls(example_id, anat, wave, mre, mre_mask, anat_mask)

    @classmethod
    def load_xarrays(cls, xarray_dir, example_id, anat=False, verbose=True):
        xarray_dir  = pathlib.Path(xarray_dir)
        example_dir = xarray_dir / str(example_id)
        if anat:
            anat = load_xarray_file(example_dir / 'anat.nc', verbose)
            anat_mask = load_xarray_file(example_dir / 'anat_mask.nc', verbose)
        else:
            anat = anat_mask = None
        wave = load_xarray_file(example_dir / 'wave.nc', verbose)
        mre  = load_xarray_file(example_dir / 'mre.nc',  verbose)
        mre_mask  = load_xarray_file(example_dir / 'mre_mask.nc',  verbose)
        return cls(example_id, anat, wave, mre, mre_mask, anat_mask)

    def save_xarrays(self, xarray_dir, verbose=True):
        xarray_dir  = pathlib.Path(xarray_dir)
        example_dir = xarray_dir / str(self.example_id)
        example_dir.mkdir(parents=True, exist_ok=True)
        if self.anat is not None:
            save_xarray_file(example_dir / 'anat.nc', self.anat, verbose)
            save_xarray_file(example_dir / 'anat_mask.nc', self.anat_mask, verbose)
        save_xarray_file(example_dir / 'wave.nc', self.wave, verbose)
        save_xarray_file(example_dir / 'mre.nc',  self.mre,  verbose)
        save_xarray_file(example_dir / 'mre_mask.nc',  self.mre_mask,  verbose)

    @property
    def metadata(self):
        return self.anat.shape, self.wave.shape, self.mre.shape

    def eval_baseline(
        self, order=3, kernel_size=5, rho=1e3, frequency=60, polar=False
    ):
        u = self.wave

        # Savitsky-Golay smoothing and derivatives
        Ku = u.copy()
        Lu = u.copy()
        resolution = u.field.spatial_resolution * 1e-3
        for z in range(u.shape[2]):
            Ku[...,z] = discrete.savgol_smoothing(
                u[...,z], order=order, kernel_size=kernel_size
            )
            Lu[...,z] = discrete.savgol_laplacian(
                u[...,z], order=order, kernel_size=kernel_size
            ) / resolution[0]**2

        # algebraic Helmholtz inversion
        Mu = discrete.helmholtz_inversion(Ku, Lu, rho, frequency, polar, eps=1e-5)

        # post-processing
        Mu.values[Mu.values < 0] = 0
        a = np.array([1, 2, 3, 2, 1])
        a = np.einsum('i,j,k->ijk', a, a, a)
        Mu_median = scipy.ndimage.median_filter(Mu, footprint=a > 2)
        Mu_outliers = np.abs(Mu - Mu_median) > 1000
        Mu.values = np.where(Mu_outliers, Mu_median, Mu)
        Mu.values = scipy.ndimage.gaussian_filter(Mu, sigma=0.65, truncate=3)
        Mu.name = 'Mwave'

        # store results
        self.Kwave = Ku
        self.Lwave = Lu
        self.Mwave = Mu

    def view(self, mask=False):
        if mask:
            anat = as_xarray(self.anat * (self.anat_mask > 0), like=self.anat)
            wave = as_xarray(self.wave * (self.mre_mask > 0), like=self.wave)
            mre = as_xarray(self.mre * (self.mre_mask > 0), like=self.mre)
            anat_viewer = XArrayViewer(anat)
            wave_viewer = XArrayViewer(wave)
            mre_viewer = XArrayViewer(mre)
            if hasattr(self, 'Mwave'):
                Mwave = as_xarray(self.Mwave * (self.mre_mask > 0), like=self.mre)
                base_viewer = XArrayViewer(Mwave)
        else:
            anat_viewer = XArrayViewer(self.anat)
            wave_viewer = XArrayViewer(self.wave)
            mre_viewer = XArrayViewer(self.mre)
            if hasattr(self, 'Mwave'):
                base_viewer = XArrayViewer(self.Mwave)


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

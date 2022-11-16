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
        pass # TODO

    @classmethod
    def from_cohort(cls, cohort):
        examples = {}
        example_ids = []
        for patient_id in cohort.patient_ids:
            patient = cohort.patients[patient_id]
            ex = MREExample.from_patient(patient)
            example_ids.append(patient_id)
            examples[patient_id] = ex
        return cls(example_ids, examples)

    @classmethod
    def from_xarrays(cls, xarray_dir, verbose=True):
        examples = {}
        example_ids = []
        for example_id in os.listdir(xarray_dir):
            ex = MREExample.from_xarrays(xarray_dir, example_id)
            example_ids.append(example_id)
            examples[example_id] = ex
        return cls(example_ids, examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if is_iterable(idx, string_ok=False) or isinstance(idx, slice):
            example_ids = self.example_ids[idx]
            examples = [self.examples[xid] for xid in example_ids]
            return type(self)(example_ids, examples)
        else:
            return self.examples[self.example_ids[idx]]

    def save_xarrays(self, xarray_dir, verbose=True):
        for xid in self.example_ids:
            ex = self.examples[xid]
            ex.save_xarrays(xarray_dir, verbose)

    def eval_baseline(self):
        for xid in self.example_ids:
            ex = self.examples[xid]
            ex.eval_baseline()

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
    def from_xarrays(cls, xarray_dir, example_id, verbose=True):
        xarray_dir = pathlib.Path(xarray_dir)
        example_dir = xarray_dir / str(example_id)
        anat = load_xarray_file(example_dir / 'anat.nc', verbose)
        wave = load_xarray_file(example_dir / 'wave.nc', verbose)
        mre = load_xarray_file(example_dir / 'mre.nc', verbose)
        mre_mask = load_xarray_file(example_dir / 'mre_mask.nc', verbose)
        anat_mask = load_xarray_file(example_dir / 'anat_mask.nc', verbose)
        return cls(example_id, anat, wave, mre, mre_mask, anat_mask)

    @classmethod
    def from_bioqic(self, bioqic):
        pass # TODO

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

    @property
    def metadata(self):
        return self.anat.shape, self.wave.shape, self.mre.shape

    def save_xarrays(self, xarray_dir, verbose=True):
        xarray_dir = pathlib.Path(xarray_dir)
        example_dir = xarray_dir / str(self.example_id)
        example_dir.mkdir(parents=True, exist_ok=True)
        save_xarray_file(example_dir / 'anat.nc', self.anat, verbose)
        save_xarray_file(example_dir / 'wave.nc', self.wave, verbose)
        save_xarray_file(example_dir / 'mre.nc', self.mre, verbose)
        save_xarray_file(example_dir / 'mre_mask.nc', self.mre_mask, verbose)
        save_xarray_file(example_dir / 'anat_mask.nc', self.anat_mask, verbose)

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
            anat = as_xarray(self.anat * self.anat_mask, like=self.anat)
            wave = as_xarray(self.wave * self.mre_mask, like=self.wave)
            mre = as_xarray(self.mre * self.mre_mask, like=self.mre)
            anat_viewer = XArrayViewer(anat)
            wave_viewer = XArrayViewer(wave)
            mre_viewer = XArrayViewer(mre)
            if hasattr(self, 'Mwave'):
                Mwave = as_xarray(self.Mwave * self.mre_mask, like=self.mre)
                base_viewer = XArrayViewer(Mwave)
        else:
            anat_viewer = XArrayViewer(self.anat)
            wave_viewer = XArrayViewer(self.wave)
            mre_viewer = XArrayViewer(self.mre)
            if hasattr(self, 'Mwave'):
                base_viewer = XArrayViewer(self.Mwave)


def save_xarray_file(nc_file, array, verbose=True):
    print_if(verbose, f'Writing {nc_file}')
    array.to_netcdf(nc_file)


def load_xarray_file(nc_file, verbose=True):
    print_if(verbose, f'Loading {nc_file}')
    return xr.open_dataarray(nc_file)

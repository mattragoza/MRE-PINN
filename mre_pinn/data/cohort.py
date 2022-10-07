import sys, pathlib, glob
import pandas as pd

from .patient import Patient, SEQUENCES
from ..utils import print_if, progress


class PatientCohort(object):
    '''
    An object for loading and preprocessing MRI and MRE
    images for a group of patients.

    Args:
        nifti_dirs: Directories to look for NIFTI files.
        patient_ids: Either a list of requested patient IDs,
            or a glob pattern for finding available patients IDs.
        sequences: Either a list of requested sequence names,
            or a glob pattern for finding available sequences.
        xarray_dir: Directory to save and load xarray files.
        verbose: If True, print verbose output.
    '''
    def __init__(
        self,
        nifti_dirs='/ocean/projects/asc170022p/shared/Data/MRE/*/NIFTI',
        patient_ids='*',
        sequences=SEQUENCES,
        xarray_dir='data/NAFLD',
        verbose=True
    ):
        self.verbose = verbose

        if isinstance(nifti_dirs, str):
            nifti_dirs = glob.glob(nifti_dirs)
        assert nifti_dirs, 'No matching NIFTI dirs'
        self.nifti_dirs = sorted([pathlib.Path(d) for d in nifti_dirs])

        if isinstance(patient_ids, str): # glob pattern
            pattern = patient_ids
            found_patients, found_ids = self.find_patients(pattern, sequences)
            assert found_patients, \
                f'No patients matching {repr(pattern)} with sequences {sequences}'
            self.patient_ids = found_ids
            self.patients = found_patients

        else: # list of requested patient ids
            patients, missing_ids = self.get_patients(patient_ids, sequences)
            assert not missing_ids, \
                f'Patients {missing_ids} are missing or missing sequences'
            self.patient_ids = patient_ids
            self.patients = patients

        self.xarray_dir = pathlib.Path(xarray_dir)

    def find_patients(self, pattern='*', sequences='*'):
        '''
        Find patients that match a glob pattern
        and have the requested imaging sequences.
        '''
        patients, patient_ids = {}, []
        for nifti_dir in self.nifti_dirs:
            for patient_dir in sorted(nifti_dir.glob(pattern)):
                pid = patient_dir.stem
                try:
                    patient = Patient(nifti_dir, pid, sequences)
                    patient_ids.append(pid)
                    patients[pid] = patient
                except AssertionError as e:
                    print_if(self.verbose, e)
                    continue
        return patients, patient_ids

    def get_patients(self, patient_ids, sequences='*'):
        '''
        Get patients from a list of requested IDs and
        imaging sequences. Returns the found patients
        the subset of patient IDs that were not found.
        '''
        requested_ids = set(patient_ids)
        found_patients, found_ids = self.find_patients('*', sequences)
        patients = {
            pid: p for pid, p in found_patients.items() if pid in requested_ids
        }
        missing_ids = requested_ids - set(found_ids) 
        return patients, missing_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        return self.patients[self.patient_ids[idx]]

    @property
    def metadata(self):
        dfs = []
        for pid in self.patient_ids:
            df = self.patients[pid].metadata
            df['patient_id'] = pid
            dfs.append(df)
        df = pd.concat(dfs).reset_index()
        return df.set_index(['patient_id', 'sequence', 'dimension'])

    def describe(self):
        dfs = []
        for pid in progress(self.patient_ids):
            df = self.patients[pid].describe()
            df['patient_id'] = pid
            dfs.append(df)
        return pd.concat(dfs)

    def load_images(self):
        for pid in progress(self.patient_ids):
            self.patients[pid].load_images()

    def preprocess_images(self):
        model = load_segment_model('cuda', verbose=self.verbose)
        for pid in progress(self.patient_ids):
            self.patients[pid].preprocess_images(model=model)

    def convert_images(self):
        for pid in progress(self.patient_ids):
            self.patients[pid].convert_images()

    def save_xarrays(self):
        for pid in progress(self.patient_ids):
            self.patients[pid].save_xarrays()

    def create_xarrays(self):
        model = load_segment_model('cuda')
        for pid in progress(self.patient_ids):
            patient = self.patients[pid]
            patient.load_images()
            patient.preprocess_images(model=model)
            patient.convert_images()
            patient.save_xarrays()

    def load_xarrays(self):
        for pid in progress(self.patient_ids):
             self.patients[pid].load_xarrays()

    def stack_xarrays(self):
        arrays = []
        for pid in self.patient_ids:
            array = self.patients[pid].stack_xarrays()
            arrays.append(array)
        dim = xr.DataArray(self.patient_ids, dims=['patient_id'])
        array = xr.concat(arrays, dim=dim)
        return array.transpose('patient_id', 'sequence', 'x', 'y', 'z')

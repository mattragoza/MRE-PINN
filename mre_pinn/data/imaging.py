import sys, pathlib, glob, functools, collections
import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage
import skimage
import SimpleITK as sitk
import torch

from .dataset import MREDataset
from .segment import UNet3D
from ..utils import print_if, progress, braced_glob, as_path_list
from ..visual import XArrayViewer
from .. import discrete


SEQUENCES = [
    't1_pre_in',
    't1_pre_water',
    't1_pre_out',
    't1_pre_fat',
    't2',
    'mre_raw',
    'wave',
    'mre'
]


class ImagingCohort(object):
    '''
    An object for loading and preprocessing MRI and MRE
    images for a group of patients.

    Args:
        nifti_dirs: Directories to look for NIFTI files.
        patient_ids: Either a list of requested patient IDs,
            or a glob pattern for finding available patients IDs.
        sequences: Either a list of requested sequence names,
            or a glob pattern for finding available sequences.
        verbose: If True, print verbose output.
    '''
    def __init__(
        self,
        patient_ids='*',
        sequences=SEQUENCES,
        nifti_dirs='/ocean/projects/asc170022p/shared/Data/MRE/*/NIFTI',
        verbose=True
    ):
        self.verbose = verbose

        if isinstance(nifti_dirs, str):
            nifti_dirs = braced_glob(nifti_dirs)
            assert nifti_dirs, 'No matching NIFTI dirs'
        else:
            nifti_dirs = as_path_list(nifti_dirs)
            missing_dirs = {d for d in nifti_dirs if not d.exists()}
            assert not missing_dirs, \
                f'NIFTI dirs {sorted(missing_dirs)} do not exist'
        self.nifti_dirs = nifti_dirs

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
                f'Patients {sorted(missing_ids)} are missing or missing sequences'
            self.patient_ids = patient_ids
            self.patients = patients

    def find_patients(self, pattern='*', sequences='*'):
        '''
        Find patients that match a glob pattern
        and have the requested imaging sequences.
        '''
        patients, patient_ids = {}, []
        for nifti_dir in self.nifti_dirs:
            for patient_dir in braced_glob(nifti_dir / pattern):
                pid = patient_dir.stem
                try:
                    patient = ImagingPatient(pid, sequences, nifti_dir)
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
        if len(patient_ids) > 1:
            pattern = '{' + ','.join(patient_ids) + '}'
        else:
            pattern = patient_ids[0]
        found_patients, found_ids = self.find_patients(pattern, sequences)
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

    def preprocess(self, **kwargs):
        model = load_segment_model('cuda', verbose=self.verbose)
        for pid in progress(self.patient_ids):
            self.patients[pid].preprocess(model=model, **kwargs)

    def to_dataset(self):
        return MREDataset.from_cohort(self)


class ImagingPatient(object):
    '''
    An object for loading and preprocessing MRI and MRE
    images for a single patient.

    Args:
        nifti_dir: Directory to look for NIFTI files.
        patient_id: Patient ID/subdirectory to locate files.
        sequences: Either a list of requested sequence names,
            or a glob pattern for finding available sequences.
        verbose: If True, print verbose output.
    '''
    def __init__(
        self,
        patient_id='0006',
        sequences=SEQUENCES,
        nifti_dir='/ocean/projects/asc170022p/shared/Data/MRE/MRE_DICOM_7-31-19/NIFTI',
        verbose=True
    ):
        self.nifti_dir = pathlib.Path(nifti_dir)
        self.patient_id = patient_id
        patient_dir = self.nifti_dir / patient_id

        if isinstance(sequences, str): # glob pattern
            pattern = sequences
            found_sequences = self.find_sequences(pattern)
            assert found_sequences, \
                f'{patient_dir} has no sequences matching {repr(sequences)}'
            self.sequences = found_sequences

        else: # list of requested sequences
            missing_sequences = self.missing_sequences(sequences)
            assert not missing_sequences, \
                f'{patient_dir} is missing sequences {missing_sequences}'
            self.sequences = sequences

        self.verbose = verbose

    def find_sequences(self, pattern='*'):
        '''
        Return a list of available imaging sequences
        for the patient that match a glob pattern.
        '''
        patient_dir = self.nifti_dir / self.patient_id
        return sorted(s.stem for s in patient_dir.glob(pattern + '.nii'))

    def missing_sequences(self, sequences):
        '''
        Return the subset of requested imaging sequences
        that are missing for the patient.
        '''
        available_sequences = self.find_sequences()
        return set(sequences) - set(available_sequences)

    def load_images(self):
        self.images = {}
        for seq in self.sequences:
            nii_file = self.nifti_dir / self.patient_id / (seq + '.nii')
            image = load_nifti_file(nii_file, self.verbose)
            image.SetMetaData('name', seq)
            self.images[seq] = image

    @property
    def metadata(self):
        index_cols = ['sequence', 'dimension']
        df = pd.DataFrame(columns=index_cols).set_index(index_cols)
        for seq, image in self.images.items():
            size = image.GetSize()
            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            for dim in range(image.GetDimension()):
                df.loc[(seq, dim), 'size'] = image.GetSize()[dim]
                df.loc[(seq, dim), 'spacing'] = image.GetSpacing()[dim]
                df.loc[(seq, dim), 'origin'] = image.GetOrigin()[dim]
        df['size'] = df['size'].astype(int)
        df['limit'] = df['origin'] + (df['size'] - 1) * df['spacing']
        df['center'] = df['origin'] + (df['size'] - 1) / 2 * df['spacing']
        df['extent'] = df['limit'] - df['origin'] + df['spacing']
        return df

    def describe(self):
        index_cols = ['sequence']
        df = pd.DataFrame(columns=index_cols).set_index(index_cols)
        for seq, image in self.images.items():
            array = sitk.GetArrayViewFromImage(image)
            df.loc[seq, 'dtype'] = array.dtype
            df.loc[seq, 'count'] = array.size
            df.loc[seq, 'mean'] = array.mean()
            df.loc[seq, 'std'] = array.std()
            df.loc[seq, 'min'] = array.min()
            df.loc[seq, '25%'] = np.percentile(array, 25)
            df.loc[seq, '50%'] = np.percentile(array, 50)
            df.loc[seq, '75%'] = np.percentile(array, 75)
            df.loc[seq, 'max'] = array.max()
        df['count'] = df['count'].astype(int)
        return df

    def preprocess(
        self,
        wave_vmax=1e-1,
        same_grid=False,
        anat_size=(256, 256, 16),
        mre_size=(256, 256, 4),
        model=None
    ):
        # The order of operations warrants some explanation.
        mre_sequences = ['mre_raw', 'wave', 'mre']
        anat_sequences = [s for s in self.sequences if s not in mre_sequences]

        # Some of the MRE images have incorrect spatial metadata,
        #   so we correct them by extrapolating from the metadata
        #   of the 'mre' sequence, which seems to be correct. We
        #   assume that the MRE images all occupy the same spatial
        #   domain, i.e. same center and overall spatial extent.
        self.correct_metadata(['mre_raw', 'wave'], using='mre')

        # The wave images we have are screenshots of some kind,
        #   so they are RGB images with text overlays. We convert
        #   the RGB pixels to grayscale using prior knowledge about
        #   the wave image colormap (we don't know the scale...)
        self.restore_wave_image('wave', vmax=wave_vmax)

        # The segmentation model accepts input size of (256, 256, 32),
        #   so we resize the main anatomic image to that size before
        #   providing it as input to the segmentation model. First, we
        #   register the main anatomic image to mre_raw so that the 
        #   mask is aligned and we can then aligned the other images.
        main_anat_seq = 't1_pre_out'
        self.register_images([main_anat_seq], fixed='mre_raw', resize=same_grid)
        self.resize_images([main_anat_seq], size=(256, 256, 32))
        self.segment_image(main_anat_seq, model=model)
        self.register_images(['mre_mask'], fixed='mre_raw', resize=True)
        self.sequences = self.sequences + ['anat_mask', 'mre_mask']

        # Then we register all of the remaining anatomic images to the
        #   main anatomic image, so all images should now be aligned.
        self.register_images(anat_sequences, fixed=main_anat_seq, resize=True)

        self.resize_images(anat_sequences + ['anat_mask'], size=anat_size)
        self.resize_images(mre_sequences + ['mre_mask'], size=mre_size)

        self.correct_metadata(anat_sequences + ['anat_mask'], using='t1_pre_in')
        self.correct_metadata(mre_sequences + ['mre_mask'], using='mre')

    def correct_metadata(self, on, using):
        ref_image = self.images[using]
        for seq in on:
            correct_metadata(self.images[seq], ref_image, self.verbose)

    def restore_wave_image(self, wave_seq, vmax):
        self.images[wave_seq] = restore_wave_image(
            self.images[wave_seq], vmax, self.verbose
        )

    def resize_images(self, sequences, size):
        for seq in sequences:
            self.images[seq] = resize_image(self.images[seq], size, self.verbose)

    def segment_image(self, input_seq, model):
        mask = segment_image(
            self.images[input_seq], model, self.verbose
        )
        self.images['anat_mask'] = mask
        self.images['mre_mask'] = mask

    def register_images(self, moving, fixed, resize=False):
        fixed_image = self.images[fixed]
        transforms = []
        for moving_seq in moving:
            moving_image = self.images[moving_seq]
            self.images[moving_seq], transform = register_image(
                moving_image, fixed_image,
                transform='rigid', resize=resize, verbose=self.verbose
            )
            transforms.append(transform)
        return transforms

    def transform_image(self, seq, transform):
        self.images[seq] = transform_image(self.images[seq], transform, self.verbose)

    def convert_images(self):
        xarrays = {}
        for seq, image in self.images.items():
            xarrays[seq] = convert_to_xarray(image, self.verbose)
        return xarrays

    def stack_xarrays(self, sequences, normalize=False, downsample=1):
        arrays = []
        for seq in sequences:
            array = self.arrays[seq]
            if normalize:
                a_min = np.percentile(array, 1)
                a_max = np.percentile(array, 99)
                a_range = a_max - a_min
                array = (array - a_min) / a_range
                if seq == 'wave':
                    array = array * 2 - 1
                if downsample > 1:
                    array = array.coarsen(
                        x=downsample, y=downsample, z=downsample
                    ).mean()
            arrays.append(array)
        dim = xr.DataArray(sequences, dims=['sequence'])
        array = xr.concat(arrays, dim=dim)
        return array.transpose('sequence', 'x', 'y', 'z')

    def to_dataset(self):
        examples = {}
        example_ids = []
        for patient_id in cohort.patient_ids:
            patient = cohort.patients[patient_id]
            ex = MREExample.from_patient(patient)
            example_ids.append(patient_id)
            examples[patient_id] = ex
        return cls(example_ids, examples)


def load_nifti_file(nii_file, verbose=True):
    print_if(verbose, f'Loading {nii_file}')
    image = sitk.ReadImage(str(nii_file))
    return image


def correct_metadata(image, ref_image, verbose=True, center=True, scale=True):
    print_if(verbose, f'Correcting metadata on {image.GetMetaData("name")}')

    # set metadata based on reference image
    # i.e. assume that wave image occupies same
    #   spatial domain as reference image
    im_size = np.array(image.GetSize())
    im_spacing = np.array(image.GetSpacing())
    ref_size = np.array(ref_image.GetSize())
    ref_spacing = np.array(ref_image.GetSpacing())
    if scale: # adjust spacing
        im_spacing = ref_spacing * ref_size / im_size
        image.SetSpacing(im_spacing)

    if center: # adjust origin
        im_origin = np.array(image.GetOrigin())
        ref_origin = np.array(ref_image.GetOrigin())
        ref_center = ref_origin + (ref_size - 1) / 2 * ref_spacing
        im_center = im_origin + (im_size - 1) / 2 * im_spacing
        im_origin += ref_center - im_center
        image.SetOrigin(im_origin)


def restore_wave_image(wave_image, vmax, verbose=True):
    print_if(verbose, f'Restoring {wave_image.GetMetaData("name")}')

    array = sitk.GetArrayViewFromImage(wave_image)
    
    if wave_image.GetNumberOfComponentsPerPixel() == 3: # convert RGB to grayscale
        array_r = array[...,0].astype(float)
        array_g = array[...,1].astype(float)
        array_b = array[...,2].astype(float)
        array_gr = np.where(array_r == 0, 0, array_g)
        array_gb = np.where(array_b == 0, 0, array_g)
        array = (array_r + array_gr) / 512 - (array_b + array_gb) / 512
        array *= vmax

        # apply inpainting to remove text
        array_txt = np.where(
            (array_r == 255) & (array_g == 255) & (array_b == 255), 1, 0
        )
        for i in range(array.shape[0]):
            array_txt[i] = skimage.morphology.binary_dilation(array_txt[i])
            array_txt[i] = skimage.morphology.binary_dilation(array_txt[i])
            array[i] = skimage.restoration.inpaint_biharmonic(array[i], array_txt[i])

    restored_image = sitk.GetImageFromArray(array)
    restored_image.CopyInformation(wave_image)
    restored_image.SetMetaData('name', wave_image.GetMetaData('name'))
    return restored_image


def resize_image(image, out_size, verbose=True):
    im_name = image.GetMetaData("name")
    print_if(verbose, f'Resizing {im_name} to {out_size}')

    in_size = np.array(image.GetSize())
    in_origin = np.array(image.GetOrigin())
    in_spacing = np.array(image.GetSpacing())
    in_center = in_origin + (in_size - 1) / 2 * in_spacing

    out_size = np.array(out_size)
    out_center = in_center
    out_spacing = in_size / out_size * in_spacing
    out_origin = out_center - (out_size - 1) / 2  * out_spacing

    ref_image = sitk.GetImageFromArray(np.ones(out_size).T)
    ref_image.SetSpacing(out_spacing)
    ref_image.SetOrigin(out_origin)

    transform = sitk.AffineTransform(3)
    if 'mask' in im_name:
        interp_method = sitk.sitkNearestNeighbor
    else:
        interp_method = sitk.sitkLinear
    resized_image = sitk.Resample(image, ref_image, transform, interp_method)
    resized_image.SetMetaData('name', image.GetMetaData('name'))
    return resized_image


def register_image(
    moving_image, fixed_image, transform='rigid', resize=False, verbose=True
):
    if verbose:
        moving_name = moving_image.GetMetaData('name')
        fixed_name = fixed_image.GetMetaData('name')
        print(f'Registering {moving_name} to {fixed_name}')

    # perform registration
    reg_params = sitk.GetDefaultParameterMap(transform)
    if 'mask' in moving_name:
        reg_params['ResampleInterpolator'] = [
            'FinalNearestNeighborInterpolator'
        ]
    reg_filter = sitk.ElastixImageFilter()
    reg_filter.SetFixedImage(fixed_image)
    reg_filter.SetMovingImage(moving_image)
    reg_filter.SetParameterMap(reg_params)
    reg_filter.SetLogToConsole(False)
    reg_filter.Execute()

    # get transformation parameters
    transform_params = reg_filter.GetTransformParameterMap()

    # make sure that aligned image is sampled at same resolution
    mvg_size = np.array(moving_image.GetSize())
    mvg_origin = np.array(moving_image.GetOrigin())
    mvg_spacing = np.array(moving_image.GetSpacing())
    mvg_center = mvg_origin + (mvg_size - 1) / 2 * mvg_spacing

    fix_size = np.array(fixed_image.GetSize())
    fix_origin = np.array(fixed_image.GetOrigin())
    fix_spacing = np.array(fixed_image.GetSpacing())
    fix_center = fix_origin + (fix_size - 1) / 2 * fix_spacing

    out_size = mvg_size
    out_center = fix_center
    out_spacing = mvg_spacing
    out_origin = out_center - (out_size - 1) / 2  * out_spacing

    as_string_vec = lambda x: [str(y) for y in x]
    transform_params[0]['Size'] = as_string_vec(out_size)
    transform_params[0]['Origin'] = as_string_vec(out_origin)
    transform_params[0]['Spacing'] = as_string_vec(out_spacing)
    if verbose:
        sitk.PrintParameterMap(transform_params)
        sys.stdout.flush()

    if resize:
        aligned_image = reg_filter.GetResultImage()
        aligned_image.SetMetaData('name', moving_image.GetMetaData('name'))
    else:
        aligned_image = transform_image(moving_image, transform_params, verbose)
    return aligned_image, transform_params


def transform_image(image, transform_params, verbose=True):
    im_name = image.GetMetaData("name")
    print_if(verbose, f'Transforming {im_name}')
    transform = sitk.TransformixImageFilter()
    if 'mask' in im_name:
        transform_params[0]['ResampleInterpolator'] = [
            'FinalNearestNeighborInterpolator'
        ]
    else:
        transform_params[0]['ResampleInterpolator'] = [
            'FinalBSplineInterpolator'
        ]
    transform.SetTransformParameterMap(transform_params)
    transform.SetMovingImage(image)
    transform.SetLogToConsole(False)
    transform.Execute()
    transformed_image = transform.GetResultImage()
    transformed_image.SetMetaData('name', image.GetMetaData('name'))
    return transformed_image


@functools.cache
def load_segment_model(device, verbose=True):
    print_if(verbose, 'Loading segmentation model')
    state_file = '/ocean/projects/asc170022p/bpollack/' \
        'mre_ai/data/CHAOS/trained_models/001/model_2020-09-30_11-14-20.pkl'
    with torch.no_grad():
        model = UNet3D()
        state_dict = torch.load(state_file, map_location=device)
        state_dict = collections.OrderedDict([
            (k[7:], v) for k, v in state_dict.items()
        ])
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    return model


def segment_image(image, model=None, verbose=True):
    if model is None:
        model = load_segment_model('cuda')
    print_if(verbose, f'Segmenting {image.GetMetaData("name")}')
    array = sitk.GetArrayFromImage(image)
    a_min, a_max = np.percentile(array, (0.5, 99.5))
    array = skimage.exposure.rescale_intensity(
        array, in_range=(a_min, a_max), out_range=(-1.0, 1.0)
    )
    with torch.no_grad():
        input_ = array[np.newaxis,np.newaxis,...]
        input_ = torch.as_tensor(input_, dtype=torch.float32)
        output = torch.sigmoid(model(input_))
        mask = torch.where(output > 0.5, 1, 0)
        mask = mask.detach().cpu().numpy()[0,0]

    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(image)
    mask_image.SetMetaData('name', 'mask')
    return mask_image


def convert_to_xarray(image, verbose=True):
    print_if(verbose, f'Converting {image.GetMetaData("name")} to xarray')

    dimension = image.GetDimension()
    if dimension == 3:
        dims = ['x', 'y', 'z']
    elif dimension == 2:
        dims = ['x', 'y']
    
    size = image.GetSize()
    origin = image.GetOrigin() # mm
    spacing = image.GetSpacing() # mm

    coords = {}
    for i, dim in enumerate(dims):
        coords[dim] = (origin[i] + np.arange(size[i]) * spacing[i]) * 2e-3 # mm

    n_components = image.GetNumberOfComponentsPerPixel()
    if n_components > 1:
        dims.append('component')
        coords['component'] = np.arange(n_components)
        array = sitk.GetArrayFromImage(image)
        axes = (2, 1, 0, 3)
        array = np.transpose(array, axes)
    else:
        array = sitk.GetArrayFromImage(image).T

    array = xr.DataArray(array, dims=dims, coords=coords)
    array.name = image.GetMetaData('name')
    return array

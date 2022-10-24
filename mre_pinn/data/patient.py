import sys, pathlib, functools, collections
import numpy as np
import pandas as pd
import xarray as xr
import skimage
import SimpleITK as sitk
import torch

from ..utils import print_if
from ..visual import XArrayViewer


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


class Patient(object):
    '''
    An object for loading and preprocessing MRI and MRE
    images for a single patient.

    Args:
        nifti_dir: Directory to look for NIFTI files.
        patient_id: Patient ID/subdirectory to locate files.
        sequences: Either a list of requested sequence names,
            or a glob pattern for finding available sequences.
        xarray_dir: Directory to save and load xarray files.
        verbose: If True, print verbose output.
    '''
    def __init__(
        self,
        nifti_dir='/ocean/projects/asc170022p/shared/Data/MRE/MRE_DICOM_7-31-19/NIFTI',
        patient_id='0006',
        sequences=SEQUENCES,
        xarray_dir='data/NAFLD',
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

        self.xarray_dir = pathlib.Path(xarray_dir)
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

    def preprocess_images(self, model=None):
        # The order of operations warrants some explanation.

        # Some of the MRE images have incorrect spatial metadata,
        #   so we correct them by extrapolating from the metadata
        #   of the 'mre' sequence, which seems to be correct. We
        #   assume that the MRE images all occupy the same spatial
        #   domain, i.e. same center and overall spatial extent.
        self.correct_metadata(on=['mre_raw', 'wave'], using='mre')

        # The wave images we have are screenshots of some kind,
        #   so they are RGB images with text overlays. We convert
        #   the RGB pixels to grayscale using prior knowledge about
        #   the wave image colormap (we don't know the scale...)
        self.restore_wave_image(wave_seq='wave', vmax=1e3)

        # The segmentation model accepts input size of (256, 256, 32),
        #   so we resize the images to that size. This corresponds to
        #   downsampling the MRI images and upsampling for MRE images,
        #   at least in the z dimension.
        #self.resize_images(size=(256, 256, 32))

        #self.segment_images(mask_seq='t1_pre_out', model=model)

        self.register_images(mask_seq='t1_pre_out')

        #self.resize_images((256, 256, 4))

    def correct_metadata(self, on, using):
        ref_image = self.images[using]
        for seq in on:
            correct_metadata(self.images[seq], ref_image, self.verbose)

    def restore_wave_image(self, wave_seq, vmax):
        self.images[wave_seq] = restore_wave_image(
            self.images[wave_seq], vmax, self.verbose
        )

    def resize_images(self, size):
        for seq, image in self.images.items():
            self.images[seq] = resize_image(image, size, self.verbose)

    def center_images(self):
        pass # TODO

    def segment_images(self, mask_seq, model):
        self.images['mask'] = segment_image(
            self.images[mask_seq], model, self.verbose
        )

    def register_images(self, mask_seq):
        fixed_image = self.images['mre_raw']
        for seq, moving_image in self.images.items():
            if seq in {'mre_raw', 'mre_phase', 'wave', 'mre', 'mask'}:
                continue
            transform = 'rigid'
            self.images[seq], transform_params = register_image(
                moving_image, fixed_image, transform, self.verbose
            )
            if seq == mask_seq:
                mask_params = transform_params
        if 'mask' in self.images:
            self.images['mask'] = transform_image(
                self.images['mask'], mask_params, self.verbose
            )

    def convert_images(self):
        self.arrays = {}
        for seq, image in self.images.items():
            self.arrays[seq] = convert_to_xarray(image, self.verbose)

    def stack_xarrays(self, normalize=False, downsample=1):
        arrays = []
        for seq in self.sequences:
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
        dim = xr.DataArray(self.sequences, dims=['sequence'])
        array = xr.concat(arrays, dim=dim)
        return array.transpose('sequence', 'x', 'y', 'z')

    def save_xarrays(self):
        self.convert_images()
        patient_dir = self.xarray_dir / self.patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        for seq, array in self.arrays.items():
            nc_file = patient_dir / (seq + '.nc')
            save_xarray_file(nc_file, array, self.verbose)

    def load_xarrays(self):
        self.arrays = {}
        for seq in self.sequences + ['mask']:
            nc_file = self.xarray_dir / self.patient_id / (seq + '.nc')
            self.arrays[seq] = load_xarray_file(nc_file, self.verbose)

    def view(self, sequences=None, compare=False, downsample=1):
        self.convert_images()
        if compare:
            array = self.stack_xarrays(normalize=True, downsample=downsample)
            viewer = XArrayViewer(array)
        else:
            for seq in sequences or self.sequences:
                viewer = XArrayViewer(self.arrays[seq])


def save_xarray_file(nc_file, array, verbose=True):
    print_if(verbose, f'Writing {nc_file}')
    array.to_netcdf(nc_file)


def load_xarray_file(nc_file, verbose=True):
    print_if(verbose, f'Loading {nc_file}')
    return xr.open_dataarray(nc_file)


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


def register_image(moving_image, fixed_image, transform='rigid', verbose=True):
    if verbose:
        moving_name = moving_image.GetMetaData('name')
        fixed_name = fixed_image.GetMetaData('name')
        print(f'Registering {moving_name} to {fixed_name}')

    # perform registration
    reg_params = sitk.GetDefaultParameterMap(transform)
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

    #aligned_image = reg_filter.GetResultImage()
    #aligned_image.SetMetaData('name', moving_image.GetMetaData('name'))
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
    from mre_ai.pytorch_arch_models_genesis import UNet3D
    state_file = '/ocean/projects/asc170022p/bpollack/mre_ai/data/CHAOS/trained_models/001/model_2020-09-30_11-14-20.pkl'
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
    origin = image.GetOrigin()
    spacing = image.GetSpacing()

    coords = {}
    for i, dim in enumerate(dims):
        coords[dim] = origin[i] + np.arange(size[i]) * spacing[i]

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

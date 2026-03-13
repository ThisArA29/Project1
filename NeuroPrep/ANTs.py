import ants
import SimpleITK as sitk
from antspynet.utilities import brain_extraction

def convert_to_DICOM(
        raw_img_path
):
    raw_img_sitk = sitk.ReadImage(raw_img_path, sitk.sitkFloat32)
    raw_img_sitk = sitk.DICOMOrient(raw_img_sitk,'RPS')

    return raw_img_sitk

def bias_correction(
        raw_img_sitk,
        ShrinkFactor = 2
):
    # Create head mask
    transformed = sitk.RescaleIntensity(raw_img_sitk, 0, 255)
    transformed = sitk.LiThreshold(transformed, 0, 1)
    head_mask = transformed

    # bias correction
    inputImage = raw_img_sitk
    inputImage = sitk.Shrink(raw_img_sitk,[ShrinkFactor] * inputImage.GetDimension())
    maskImage = sitk.Shrink(head_mask, [ShrinkFactor] * inputImage.GetDimension())

    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = bias_corrector.Execute(inputImage, maskImage)
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)
    corrected_image_full_resolution = raw_img_sitk / sitk.Exp( log_bias_field )

    return corrected_image_full_resolution

def skull_stripping(
        raw_img_ants,
        modality = "t1"
):
    prob_brain_mask = brain_extraction(
        raw_img_ants,
        modality,
        verbose = True
    )

    brain_mask = ants.get_mask(prob_brain_mask, low_thresh = 0.5)
    masked = ants.mask_image(raw_img_ants, brain_mask)

    return brain_mask, masked

def intensity_normalization(
        raw_img_path,
        template_img_path
):
    raw_img_sitk = convert_to_DICOM(raw_img_path)
    template_img_sitk = convert_to_DICOM(template_img_path)

    transformed = sitk.HistogramMatching(raw_img_sitk, template_img_sitk)

    return transformed

def linear_registration(
        template_img_path,
        raw_img_ants_path
):
    template_img_ants = ants.image_read(template_img_path, reorient = "IAL")
    raw_img_ants = ants.image_read(raw_img_ants_path, reorient = "IAL")

    transformation = ants.registration(
        fixed = template_img_ants,
        moving = raw_img_ants, 
        type_of_transform = 'SyN',
        verbose = True
    )
    registered_img_ants = transformation['warpedmovout']

    return registered_img_ants
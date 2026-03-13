import subprocess
import os

def check_scans(
        scan
):
    header = scan.header

    return header

def bias_field_correction(
        input_image,
        output_name
):
    '''
    Corrects intensity variations in MRI images caused by the bias field/ magnetic field,
    which can be created by imperfections in the image acquisition process and patient
    anatomy.
    '''
    print("input name", input_image)
    command = [
        "fast", "-B",
        "-o" , output_name,
        input_image
    ]

    subprocess.run(command, check = True)

    output_corrected = f"{output_name}_restore.nii.gz"
    if not os.path.exists(output_corrected):
        raise RuntimeError("Bias Field Correction Failed.")
    print("output", output_corrected)
    return output_corrected

def skull_stripping(
        input_image,
        output_name,
        f
):
    '''
    Extracting the brain area from the MRI scan
    '''
    print("input name", input_image)
    command = [
        "bet",
        input_image,
        output_name,
        "-f", str(f), "-g", str(0), "-m"
    ]
    # command = [
    #     "bet",
    #     input_image,
    #     output_name,
    #     "-f", str(0.5), "-g", str(0), "-R", "-B"
    # ]

    subprocess.run(command, check = True)
    output_corrected = f"{output_name}.nii.gz"
    if not os.path.exists(output_corrected):
        raise RuntimeError("Skull stripping failed.")
    print("output", output_corrected)
    return output_corrected

def intensity_normalization(
        input_image,
        output_name
):
    '''
    Scaling or transforming image intensities so that the images are consistent across
    different scans or subjects.
    '''
    print("input name", input_image)
    mean = float(subprocess.check_output([
        "fslstats",
        input_image,
        "-M"
    ]).strip())

    std = float(subprocess.check_output([
        "fslstats",
        input_image,
        "-S"
    ]).strip())

    command = [
        "fslmaths",
        input_image,
        "-sub", str(mean), "-div", str(std),
        output_name
    ]

    subprocess.run(command, check = True)
    output_corrected = f"{output_name}.nii.gz"
    if not os.path.exists(output_corrected):
        raise RuntimeError("Intensity normalization failed.")
    print("output", output_corrected)
    return output_corrected

def motion_correction(
        input_image,
        output_name
):
    '''
    Remove artifacts caused by motion during image acquisition.
    '''
    print("input name", input_image)
    command = [
        "mcflirt",  # Motion correction
        "-in", input_image, # input image
        "-out", output_name # output file name
    ]

    subprocess.run(command, check = True)

    output_corrected = f"{output_name}.nii.gz"
    if not os.path.exists(output_corrected):
        raise RuntimeError("Motion correction failed.")
    print("output", output_corrected)
    return output_corrected

def linear_registration(
        input_image,
        reference_image,
        dof,
        output_name
):
    '''
    Transforming an individual brain image into a standard linear coordinate space
    (e.g. MNI152 space, Talairach space) so that brain regions across different
    subjects can be directly compared.
    '''
    print("input name", input_image)
    command = [
        "flirt",  
        "-in", input_image, 
        "-ref", reference_image,
        "-dof", dof,
        "-out", output_name, 
        "-omat", "test_img/matrix" 
    ]

    subprocess.run(
        command, 
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL
    )
    
    output_corrected = f"{output_name}.nii.gz"
    if not os.path.exists(output_corrected):
        raise RuntimeError("Spatial normalization failed.")
    print("output", output_corrected)

    return output_corrected
from ipywidgets import interact, IntSlider, Dropdown
import numpy as np
import cv2
import matplotlib.pyplot as plt

def explore_3D_array(
        arr: np.ndarray, 
        cmap: str = 'gray'
):
    """
    Given a 3D array with shape (Z,X,Y) This function will create an interactive
    widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. 
    The purpose of this function to visual inspect the 2D arrays in the image. 

    Args:
        arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
        cmap : Which color map use to plot the slices in matplotlib.pyplot
    """

    def view_slice(axis='axial', slice_idx=0):
        if axis == 'axial':        # Z
            img = arr[slice_idx, :, :]
        elif axis == 'coronal':    # Y
            img = arr[:, slice_idx, :]
        elif axis == 'sagittal':   # X
            img = arr[:, :, slice_idx]
        else:
            raise ValueError("Invalid axis")

        plt.figure(figsize = (6, 6))
        plt.imshow(img.T, cmap = cmap, origin = 'lower')  # Transpose for correct orientation
        plt.title(f"{axis.capitalize()} slice {slice_idx}")
        plt.axis("off")
        plt.show()

    def interactive_view(axis='axial'):
        max_index = {
            'axial': arr.shape[0] - 1,
            'coronal': arr.shape[1] - 1,
            'sagittal': arr.shape[2] - 1
        }[axis]

        return interact(
            view_slice, 
            axis = Dropdown(
                options=['axial', 'coronal', 'sagittal'], 
                value=axis
            ),
            slice_idx = IntSlider(
                min = 0, 
                max = max_index, 
                step = 1
            )
        )

    interactive_view()

def explore_3D_array_comparison(
        arr_before: np.ndarray, 
        arr_after: np.ndarray, 
        cmap: str = 'gray'
):
    """
    Given two 3D arrays with shape (Z,X,Y) This function will create an interactive
    widget to check out all the 2D arrays with shape (X,Y) inside the 3D arrays.
    The purpose of this function to visual compare the 2D arrays after some transformation. 

    Args:
        arr_before : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, before any transform
        arr_after : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, after some transform    
        cmap : Which color map use to plot the slices in matplotlib.pyplot
    """

    assert arr_after.shape == arr_before.shape

    def view_slice(axis = 'axial', slice_idx = 0):
        if axis == 'axial':        # Z axis
            img_before = arr_before[slice_idx, :, :]
            img_after = arr_after[slice_idx, :, :]
        elif axis == 'coronal':    # Y axis
            img_before = arr_before[:, slice_idx, :]
            img_after = arr_after[:, slice_idx, :]
        elif axis == 'sagittal':   # X axis
            img_before = arr_before[:, :, slice_idx]
            img_after = arr_after[:, :, slice_idx]
        else:
            raise ValueError("Invalid axis")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
        ax1.imshow(img_before.T, cmap=cmap, origin='lower')
        ax1.set_title('Before')
        ax1.axis("off")

        ax2.imshow(img_after.T, cmap = cmap, origin = 'lower')
        ax2.set_title('After')
        ax2.axis("off")

        plt.suptitle(f"{axis.capitalize()} slice {slice_idx}", fontsize=14)
        plt.tight_layout()
        plt.show()

    def interactive_view(axis = 'axial'):
        max_index = {
            'axial': arr_before.shape[0] - 1,
            'coronal': arr_before.shape[1] - 1,
            'sagittal': arr_before.shape[2] - 1
        }[axis]
        return interact(
            view_slice,
            axis = Dropdown(
                options = ['axial', 'coronal', 'sagittal'], 
                value = axis, 
                description = 'Axis'
            ),
            slice_idx = IntSlider(
                min = 0, 
                max = max_index, 
                step = 1, 
                description = 'Slice')
            )

    interactive_view()

def rescale_linear(
        array: np.ndarray, 
        new_min: int, 
        new_max: int
):
    """Rescale an array linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def explore_3D_array_with_mask_contour(
        arr: np.ndarray, 
        mask: np.ndarray, 
        thickness: int = 1
):
    """
    Given a 3D array with shape (Z,X,Y) This function will create an interactive
    widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. The binary
    mask provided will be used to overlay contours of the region of interest over the 
    array. The purpose of this function is to visual inspect the region delimited by the mask.

    Args:
        arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
        mask : binary mask to obtain the region of interest
    """
    assert arr.shape == mask.shape

    _arr = rescale_linear(arr,0,1)
    _mask = rescale_linear(mask,0,1)
    _mask = _mask.astype(np.uint8)

    def fn(axis='axial', slice_idx=0):
        if axis == 'axial':
            img = _arr[slice_idx, :, :]
            msk = _mask[slice_idx, :, :]
        elif axis == 'coronal':
            img = _arr[:, slice_idx, :]
            msk = _mask[:, slice_idx, :]
        elif axis == 'sagittal':
            img = _arr[:, :, slice_idx]
            msk = _mask[:, :, slice_idx]
        else:
            raise ValueError("Invalid axis selected")

        # img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_rgb = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_with_contours = cv2.drawContours(img_rgb.copy(), contours, -1, (0, 1, 0), thickness)

        plt.figure(figsize=(6, 6))
        plt.imshow(img_with_contours)
        plt.title(f"{axis.capitalize()} slice {slice_idx}")
        plt.axis("off")
        plt.show()

    interact(
        fn,
        axis = Dropdown(
            options = ['axial', 'coronal', 'sagittal'], 
            value = 'axial', 
            description = 'Axis'
        ),
        slice_idx = IntSlider(
            min = 0, 
            max = arr.shape[0] - 1, 
            step = 1, 
            description = 'Slice'
        )
    )

def add_suffix_to_filename(filename: str, suffix:str) -> str:
    """
    Takes a NIfTI filename and appends a suffix.

    Args:
        filename : NIfTI filename
        suffix : suffix to append

    Returns:
        str : filename after append the suffix
    """
    if filename.endswith('.nii'):
        result = filename.replace('.nii', f'_{suffix}.nii')
        return result
    elif filename.endswith('.nii.gz'):
        result = filename.replace('.nii.gz', f'_{suffix}.nii.gz')
        return result
    else:
        raise RuntimeError('filename with unknown extension')
import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.fft import fftn, ifftn, fftshift, ifftshift

def _rand_uniform(a, b):
    return np.random.uniform(a, b) if a != b else float(a)

class RandomIntensityScaleShift:
    """
    Multiply by a random scale and add a random shift.
    scale in [smin, smax], shift in [tmin, tmax].
    """

    def __init__(self, scale = (0.9, 1.1), shift = (-0.05, 0.05), p = 1):
        self.scale, self.shift, self.p = scale, shift, p

    def __call__(self, image):
        if np.random.rand() > self.p: return image
        s = _rand_uniform(*self.scale); t = _rand_uniform(*self.shift)

        return (image * s + t).astype(image.dtype, copy = False)
    
class RandomGaussianNoise:
    """Add N(0, sigma) noise; sigma sampled from [smin, smax]."""

    def __init__(self, sigma=(0.0, 0.05)):
        self.sigma = sigma

    def __call__(self, image):
        smin, smax = self.sigma
        sig = np.random.uniform(smin, smax)
        noise = np.random.normal(0.0, sig, size = image.shape)
        return (image + noise).astype(image.dtype, copy = False)
    
class RandomRicianNoise:
    def __init__(self, sigma = (0.0, 0.05), clip = None):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, image):
        smin, smax = self.sigma
        sig = np.random.uniform(smin, smax)
        n_r = np.random.normal(0.0, sig, image.shape)
        n_i = np.random.normal(0.0, sig, image.shape)
        y = np.sqrt((image + n_r) ** 2 + n_i ** 2)
        if self.clip is not None:
            y = np.clip(y, self.clip[0], self.clip[1])
        return y.astype(image.dtype, copy = False)
    
class RandomPoissonNoise:
    def __init__(self, peak = 20, p = 1, clip = True, eps = 1e-8, seed = 42):
        self.peak = peak
        self.p = p
        self.clip = clip
        self.eps = eps
        self.rng = np.random.default_rng(seed)

    def __call__(self, image, seed = 42):
        rng = np.random.default_rng(seed) if seed is not None else self.rng

        if rng.random() > self.p:
            return image

        img = image.astype(np.float32, copy=False)
        imin, imax = np.nanmin(img), np.nanmax(img)
        if not np.isfinite(imin) or not np.isfinite(imax) or (imax - imin) < self.eps:
            return image

        peak_val = (
            rng.uniform(*self.peak) if isinstance(self.peak, tuple)
            else float(self.peak))
        peak_val = max(peak_val, 1e-6)

        x01 = (img - imin) / (imax - imin + self.eps)
        lam = peak_val * x01
        counts = rng.poisson(lam).astype(np.float32)

        y01 = counts / peak_val
        y = y01 * (imax - imin) + imin

        if self.clip:
            y = np.clip(y, imin, imax)

        return y.astype(image.dtype, copy = False)
    
class RandomGibbsRinging:
    def __init__(self, truncation_range = (0.55, 0.96), p = 0.8):

        self.truncation_range, self.p = truncation_range, p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p:
            return image

        truncation_ratio = _rand_uniform(*self.truncation_range)
        
        if truncation_ratio >= 1.0:
             return image

        original_dtype = image.dtype

        k_space = fftshift(fftn(image))
        
        dims = np.array(image.shape)
        
        truncated_dims = (dims * truncation_ratio).astype(int)

        truncated_dims[truncated_dims < 1] = 1

        truncated_k_space = np.zeros_like(k_space, dtype=k_space.dtype)
        
        starts = (dims - truncated_dims) // 2
        ends = starts + truncated_dims

        truncated_k_space[
            starts[0]:ends[0], 
            starts[1]:ends[1], 
            starts[2]:ends[2]
        ] = k_space[
            starts[0]:ends[0], 
            starts[1]:ends[1], 
            starts[2]:ends[2]
        ]

        ringing_image = np.real(ifftn(ifftshift(truncated_k_space)))

        ringing_max = ringing_image.max()
        if ringing_max <= 0:
            return image

        ringing_image = ringing_image * (image.max() / ringing_max)

        return ringing_image.astype(original_dtype, copy = False)
    
class RandomizeBrainRegionVoxels():
    def __init__(
            self,
            atlas_dict
    ):
        self.atlas_dict = atlas_dict

    def _create(
            self,
            image,
            rand_type,
            binary_mask = None
    ):
        if rand_type == "similar":
            seed = 42
        elif rand_type == "different":
            seed = None
        elif rand_type == "complete":
            seed = 42
        else:
            raise ValueError("rand_type should be similar, different, or complete")

        if rand_type in ("similar", "different"):
            perm = self.build_label_permutation(seed, label_map = self.atlas_dict["data"])
            return self.apply_label_permutation(image, perm)

        if rand_type == "complete":
            if binary_mask is not None:
                return self.apply_complete_randomization_in_mask(image, binary_mask, seed = seed)
            else:
                return self.apply_complete_randomization(image, seed = seed)

    def build_label_permutation(
            self, seed, ignore_labels = (0,), label_map = None
    ):
        rng = np.random.default_rng(seed)
        labs = label_map if label_map is not None else self.atlas_dict["data"]
        flat_lab = labs.reshape(-1)

        perm = {}
        for lab in np.unique(flat_lab):
            if lab in ignore_labels:
                continue
            idx = np.where(flat_lab == lab)[0]
            if idx.size <= 1:
                perm[lab] = (idx, idx)
                continue
            src = idx.copy()
            rng.shuffle(src)
            perm[lab] = (idx, src)

        return perm

    def apply_label_permutation(
            self,
            volume, 
            perm
    ):
        out = volume.copy()

        spatial_ndim = self.atlas_dict["data"].ndim
        spatial_size = np.prod(self.atlas_dict["data"].shape, dtype = int)

        flat_out = out.reshape((spatial_size, -1))
        for lab, (tgt, src) in perm.items():
            flat_out[tgt] = flat_out[src]

        return out
    
    def apply_complete_randomization(self, volume, seed = 42):
        
        rng = np.random.default_rng(seed)
        spatial_ndim = self.atlas_dict["data"].ndim
        spatial_shape = volume.shape[:spatial_ndim]
        nvox = int(np.prod(spatial_shape))
        flat = volume.reshape((nvox, -1))
        idx = rng.permutation(nvox)
        flat_shuffled = flat[idx]

        return flat_shuffled.reshape(volume.shape)

    def apply_complete_randomization_in_mask(self, volume, binary_mask, seed = 42):
        
        rng = np.random.default_rng(seed)
        mask = (binary_mask.astype(bool))
        spatial_ndim = mask.ndim
        spatial_shape = mask.shape
        
        assert volume.shape[:spatial_ndim] == spatial_shape, "mask/volume spatial shape mismatch"

        nvox = int(np.prod(spatial_shape))
        flat = volume.reshape((nvox, -1))
        flat_mask = mask.reshape(-1)

        idx_in = np.where(flat_mask)[0]
        if idx_in.size <= 1:
            return volume

        perm_in = idx_in.copy()
        rng.shuffle(perm_in)
        flat[idx_in] = flat[perm_in]

        return flat.reshape(volume.shape)

'''
source code - https://github.com/moboehle/Pytorch-LRP
'''

class Flip:
    """
    Flip the input along a given axis.

    Arguments:
        axis: axis to flip over. Default is 0
        prob: probability to flip the image. Executes always when set to
             1. Default is 0.5
    """
    def __init__(self, axis = 0, prob = 1):
        self.axis = axis
        self.prob = prob

    def __call__(self, image):
        rand = np.random.uniform()
        if rand <= self.prob:
            augmented = np.flip(image, axis=self.axis).copy()
        else:
            augmented = image
        return augmented

class SagittalFlip(Flip):
    """
    Flip image along the sagittal axis (x-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, prob = 1):
        super().__init__(axis = 0, prob = prob)
    
    def __call__(self, image):
        assert(len(image.shape) == 3)
        return super().__call__(image)
    
class CoronalFlip(Flip):
    """
    Flip image along the coronal axis (y-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, prob = 1):
        super().__init__(axis=1, prob=prob)

    def __call__(self, image):
        assert(len(image.shape) == 3)
        return super().__call__(image)

class AxialFlip(Flip):
    """
    Flip image along the axial axis (z-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, prob = 1):
        super().__init__(axis=2, prob=prob)

    def __call__(self, image):
        assert(len(image.shape) == 3)
        return super().__call__(image)
    
class Rotate:
    """ 
    Rotate the input along a given axis.

    Arguments:
        axis: axis to rotate. Default is 0
        deg: min and max rotation angles in degrees. Randomly rotates 
            within that range. Can be scalar, list or tuple. In case of 
            scalar it rotates between -abs(deg) and abs(deg). Default is
            (-3, 3).
    """
    def __init__(self, axis = 0, deg = (-3, 3)):
        if axis == 0:
            self.axes = (1, 0)
        elif axis == 1:
            self.axes = (2, 1)
        elif axis == 2:
            self.axes = (0, 2)

        if isinstance(deg, tuple) or isinstance(deg, list):
            assert(len(deg) == 2)
            self.min_rot = np.min(deg)
            self.max_rot = np.max(deg)
        else:
            self.min_rot = -int(abs(deg))
            self.max_rot = int(abs(deg))

    def __call__(self, image):
        rand = np.random.randint(self.min_rot, self.max_rot + 1)
        augmented = rotate(
            image,
            angle=rand,
            axes=self.axes,
            reshape=False
            ).copy()
        return augmented

class SagittalRotate(Rotate):
    """
    Rotate image's sagittal axis (x-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, deg=(-3, 3)):
        super().__init__(axis=0, deg=deg)

class CoronalRotate(Rotate):
    """
    Rotate image's coronal axis (y-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, deg=(-3, 3)):
        super().__init__(axis=1, deg=deg)

class AxialRotate(Rotate):
    """
    Rotate image's axial axis (z-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, deg=(-3, 3)):
        super().__init__(axis=2, deg=deg)

class Translate:
    """
    Translate the input along a given axis.

    Arguments:
        axis: axis to rotate. Default is 0
        dist: min and max translation distance in pixels. Randomly 
            translates within that range. Can be scalar, list or tuple. 
            In case of scalar it translates between -abs(dist) and 
            abs(dist). Default is (-3, 3).
    """
    def __init__(self, axis=0, dist=(-2, 2)):
        self.axis = axis

        if isinstance(dist, tuple) or isinstance(dist, list):
            assert(len(dist) == 2)
            self.min_trans = np.min(dist)
            self.max_trans = np.max(dist)
        else:
            self.min_trans = -int(abs(dist))
            self.max_trans = int(abs(dist))

    def __call__(self, image):
        rand = np.random.randint(self.min_trans, self.max_trans + 1)
        augmented = np.zeros_like(image)
        if self.axis == 0:
            if rand < 0:
                augmented[-rand:, :] = image[:rand, :]
            elif rand > 0:
                augmented[:-rand, :] = image[rand:, :]
            else:
                augmented = image
        elif self.axis == 1:
            if rand < 0:
                augmented[:,-rand:, :] = image[:,:rand, :]
            elif rand > 0:
                augmented[:,:-rand, :] = image[:,rand:, :]
            else:
                augmented = image
        elif self.axis == 2:
            if rand < 0:
                augmented[:,:,-rand:] = image[:,:,:rand]
            elif rand > 0:
                augmented[:,:,:-rand] = image[:,:,rand:]
            else:
                augmented = image
        return augmented

class SagittalTranslate(Translate):
    """
    Translate image along the sagittal axis (x-axis).
    Expects input shape (X, Y, Z).
    """
    def __init__(self, dist=(-2, 2)):
        super().__init__(axis=0, dist=dist)

class CoronalTranslate(Translate):
    """
    Translate image along the coronal axis (y-axis).
    Expects input shape (X, Y, Z).
    """
    def __init__(self, dist=(-3, 3)):
        super().__init__(axis=1, dist=dist)

class AxialTranslate(Translate):
    """
    Translate image along the axial axis (z-axis).
    Expects input shape (X, Y, Z).
    """
    def __init__(self, dist = (-3, 3)):
        super().__init__(axis = 2, dist = dist)
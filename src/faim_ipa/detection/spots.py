from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_laplace
from skimage.morphology import h_maxima, ball
from skimage.util import img_as_float32

from faim_ipa.detection.utils import estimate_log_rescale_factor


def detect_spots(
    img: np.ndarray,
    axial_sigma: float,
    lateral_sigma: float,
    h: int,
    background_img: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Detect diffraction limited spots.

    The spot detection uses a Laplacian of Gaussian filter to detect
    spots of a given size. These detections are intensity filtered with
    a h-maxima filter.

    Parameters
    ----------
    img :
        Image containing spot signal.
    axial_sigma :
        Z extension of the spots.
    lateral_sigma :
        YX extension of the spots.
    h :
        h-maxima threshold.
    background_img :
        Estimated background image. This is subtracted before the
        spot-detection.

    Returns
    -------
    Detected spots.
    """
    if background_img is not None:
        image = img_as_float32(img) - img_as_float32(background_img)
    else:
        image = img_as_float32(img)

    rescale_factor = estimate_log_rescale_factor(
        axial_sigma=axial_sigma, lateral_sigma=lateral_sigma
    )
    log_img = (
        -gaussian_laplace(image, sigma=(axial_sigma, lateral_sigma, lateral_sigma))
        * rescale_factor
    )

    h_ = img_as_float32(np.array(h, dtype=img.dtype))
    h_detections = h_maxima(log_img, h=h_, footprint=ball(1))
    return np.array(np.where(h_detections)).T

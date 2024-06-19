from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_laplace
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima, ball
from skimage.util import img_as_float32
from skimage.feature.blob import _prune_blobs

from faim_ipa.detection.utils import estimate_log_rescale_factor


def detect_blobs(
    img: np.ndarray,
    axial_sigma: float,
    lateral_sigma: float,
    h: int,
    n_scale_levels: int,
    overlap: float,
    background_img: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Detect blobs of different sizes.

    The blob detection finds blobs of different sizes by applying a
    Laplacian of Gaussian with increasing sigmas followed by h-maxima
    filtering. The blob detection starts with the provided sigmas and
    then doubles them for `n_scale_levels`.

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
    n_scale_levels :
        Number of upscaling rounds.
    overlap :
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.
    background_img :
        Estimated background image. This is subtracted before the
        blob-detection.

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

    sigmas = [
        (axial_sigma * 2**i, lateral_sigma * 2**i, lateral_sigma * 2**i)
        for i in range(n_scale_levels)
    ]

    scale_cube = np.empty(image.shape + (len(sigmas),), dtype=np.uint8)

    h_ = img_as_float32(np.array(h, dtype=img.dtype))
    for i, sigma in enumerate(sigmas):
        log_img = (
            -gaussian_laplace(image, sigma=sigma)
            * rescale_factor
            * (np.mean(sigma) / np.mean(sigmas[0])) ** 2
        )
        scale_cube[..., i] = h_maxima(log_img, h=h_, footprint=ball(1))

    maxima = peak_local_max(
        scale_cube,
        threshold_abs=0.1,
        exclude_border=False,
        footprint=np.ones((3,) * scale_cube.ndim),
    )

    # Convert local_maxima to float64
    lm = maxima.astype(np.float64)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = np.array(sigmas)[maxima[:, -1]]

    # Remove sigma index and replace with sigmas
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]

    return _prune_blobs(np.array(lm), overlap=overlap, sigma_dim=sigma_dim)

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.ndimage import gaussian_filter, gaussian_laplace

from faim_ipa.detection.utils import (
    compute_axial_sigma,
    compute_lateral_sigma,
    estimate_log_rescale_factor,
)


def test_compute_axial_sigma():
    wl = 514
    NA = 1.45
    axial_spacing = 100

    assert_almost_equal(compute_axial_sigma(wl, NA, axial_spacing), 2.08, 2)


def test_compute_lateral_sigma():
    wl = 514
    NA = 1.45
    lateral_spacing = 100
    print(compute_lateral_sigma(wl, NA, lateral_spacing))
    assert_almost_equal(compute_lateral_sigma(wl, NA, lateral_spacing), 0.75, 2)


def test_estimate_log_rescale_factor():
    rescale_factor = estimate_log_rescale_factor(2.07, 0.75)

    # Create image with diffraction limited spot with intensity = 10
    img = np.zeros((101, 101, 101), dtype=np.float32)
    img[50, 50, 50] = 1
    img = gaussian_filter(img, (2.07, 0.75, 0.75))
    img = 10 * img / img.max()

    # Max value of the rescaled LoG filter response should be 10
    assert_almost_equal(
        np.max(-gaussian_laplace(img, (2.07, 0.75, 0.75)) * rescale_factor), 10, 2
    )

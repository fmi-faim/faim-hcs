import numpy as np
from numpy.testing import assert_array_equal
from scipy.ndimage import gaussian_filter

from faim_ipa.detection.spots import detect_spots


def test_detect_spots():
    np.random.seed(0)
    img = np.zeros((101, 101, 101), dtype=np.float32)

    # Create 3 spots with intensities of 100, 200, 300
    img[25, 25, 25] = 1
    img[50, 50, 50] = 2
    img[75, 75, 75] = 3
    img = gaussian_filter(img, (2.07, 0.75, 0.75))
    img = 300 * img / img.max()

    # Create 1 larger spot
    large_spot = np.zeros((101, 101, 101), dtype=np.float32)
    large_spot[50, 75, 75] = 1
    large_spot = gaussian_filter(large_spot, (2 * 2.07, 3 * 0.75, 3 * 0.75))
    large_spot = 100 * large_spot / large_spot.max()

    # Create random background noise
    background_img = np.random.normal(10, 2, (101, 101, 101)).astype(np.float32)

    # Create hot-pixel
    hot_pixels = np.zeros((101, 101, 101), dtype=np.float32)
    hot_pixels[25, 25, 50] = 500

    # Combine to final image
    img_final = (img + large_spot + hot_pixels + background_img).astype(np.uint16)

    # Fake estimated background with hot-pixels
    estimated_bg = (np.ones_like(background_img) * background_img.mean()).astype(
        np.uint16
    )
    estimated_bg[hot_pixels > 0] = 500

    # Detect spots without estimated background
    spots = detect_spots(
        img=img_final,
        axial_sigma=2.07,
        lateral_sigma=0.75,
        h=90,
        background_img=None,
    )
    assert spots.shape[0] == 4
    assert_array_equal(
        spots,
        np.array(
            [
                [25, 25, 25],
                [25, 25, 50],
                [50, 50, 50],
                [75, 75, 75],
            ]
        ),
    )

    # Detect spot with estimated background
    spots = detect_spots(
        img=img_final,
        background_img=estimated_bg,
        axial_sigma=2.07,
        lateral_sigma=0.75,
        h=90,
    )
    assert spots.shape[0] == 3
    assert_array_equal(
        spots,
        np.array(
            [
                [25, 25, 25],
                [50, 50, 50],
                [75, 75, 75],
            ]
        ),
    )

    # Detect spots brighter than 190
    spots = detect_spots(
        img=img_final,
        background_img=estimated_bg,
        axial_sigma=2.07,
        lateral_sigma=0.75,
        h=190,
    )
    assert spots.shape[0] == 2
    assert_array_equal(
        spots,
        np.array(
            [
                [50, 50, 50],
                [75, 75, 75],
            ]
        ),
    )

    # Detect spots brighter than 290
    spots = detect_spots(
        img=img_final,
        background_img=estimated_bg,
        axial_sigma=2.07,
        lateral_sigma=0.75,
        h=290,
    )
    assert spots.shape[0] == 1
    assert_array_equal(
        spots,
        np.array(
            [
                [75, 75, 75],
            ]
        ),
    )

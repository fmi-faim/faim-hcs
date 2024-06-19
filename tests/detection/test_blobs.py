import numpy as np
from numpy.testing import assert_array_equal
from scipy.ndimage import gaussian_filter

from faim_ipa.detection.blobs import detect_blobs


def test_detect_blobs():
    np.random.seed(0)
    img = np.zeros((101, 101, 101), dtype=np.float32)

    # Create 3 spots with intensities of 100, 200, 300
    img[25, 25, 25] = 1
    img[50, 50, 50] = 1
    img[75, 75, 75] = 1
    img = gaussian_filter(img, (2.07, 0.75, 0.75))
    img = 300 * img / img.max()

    # Create 1 larger spot
    large_spot = np.zeros((101, 101, 101), dtype=np.float32)
    large_spot[50, 75, 75] = 1
    large_spot = gaussian_filter(large_spot, (2 * 2.07, 2 * 0.75, 2 * 0.75))
    large_spot = 300 * large_spot / large_spot.max()

    # Create random background noise
    background_img = np.random.normal(10, 2, (101, 101, 101)).astype(np.float32)

    # Create hot-pixel
    hot_pixels = np.zeros((101, 101, 101), dtype=np.float32)
    hot_pixels[25, 25, 50] = 1000

    # Combine to final image
    img_final = (img + large_spot + hot_pixels + background_img).astype(np.uint16)

    # Fake estimated background with hot-pixels
    estimated_bg = (np.ones_like(background_img) * background_img.mean()).astype(
        np.uint16
    )
    estimated_bg[hot_pixels > 0] = 1000

    # Detect spots without estimated background
    blobs = detect_blobs(
        img=img_final,
        axial_sigma=2.07,
        lateral_sigma=0.75,
        h=200,
        n_scale_levels=2,
        overlap=0.875,
        background_img=None,
    )
    assert blobs.shape[0] == 5
    assert_array_equal(
        blobs,
        np.array(
            [
                [25, 25, 25, 2.07, 0.75, 0.75],
                [25, 25, 50, 2.07, 0.75, 0.75],
                [50, 50, 50, 2.07, 0.75, 0.75],
                [50, 75, 75, 2 * 2.07, 2 * 0.75, 2 * 0.75],
                [75, 75, 75, 2.07, 0.75, 0.75],
            ]
        ),
    )

    # Detect spots with estimated background
    blobs = detect_blobs(
        img=img_final,
        axial_sigma=2.07,
        lateral_sigma=0.75,
        h=200,
        n_scale_levels=2,
        overlap=0.875,
        background_img=estimated_bg,
    )
    assert blobs.shape[0] == 4
    assert_array_equal(
        blobs,
        np.array(
            [
                [25, 25, 25, 2.07, 0.75, 0.75],
                [50, 50, 50, 2.07, 0.75, 0.75],
                [50, 75, 75, 2 * 2.07, 2 * 0.75, 2 * 0.75],
                [75, 75, 75, 2.07, 0.75, 0.75],
            ]
        ),
    )

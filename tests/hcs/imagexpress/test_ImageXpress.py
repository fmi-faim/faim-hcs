from dataclasses import dataclass
from pathlib import Path

import pytest
from numpy.testing import assert_almost_equal

from faim_hcs.hcs import imagexpress
from faim_hcs.hcs.acquisition import (
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_hcs.hcs.imagexpress import (
    ImageXpressPlateAcquisition,
    SinglePlaneAcquisition,
    StackAcquisition,
)


@pytest.fixture
def acquisition_dir():
    return Path(__file__).parent.parent.parent.parent / "resources" / "Projection-Mix"


@pytest.fixture
def single_plane_acquisition(acquisition_dir):
    return SinglePlaneAcquisition(acquisition_dir, alignment=TileAlignmentOptions.GRID)


def test_single_plane_acquistion(single_plane_acquisition: PlateAcquisition):
    wells = single_plane_acquisition.get_well_acquisitions()

    assert wells is not None
    assert len(wells) == 2
    # MIPs: 1 well has 2 fields * 3 channels = 6 files
    assert len(wells[0]._files) == 6
    assert len(wells[0]._files) == 6

    channels = single_plane_acquisition.get_channel_metadata()
    assert len(channels) == 3
    ch = channels[0]
    assert ch.channel_index == 0
    assert ch.channel_name == "Maximum-Projection_FITC_05"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == 536
    assert ch.z_spacing is None

    ch = channels[1]
    assert ch.channel_index == 1
    assert ch.channel_name == "Best-Focus-Projection_FITC_05"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == 536
    assert ch.z_spacing is None

    ch = channels[2]
    assert ch.channel_index == 2
    assert ch.channel_name == "Maximum-Projection_FITC_05"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == 536
    assert ch.z_spacing is None

    for well in single_plane_acquisition.get_well_acquisitions():
        assert isinstance(well, WellAcquisition)
        assert len(well.get_tiles()) == 6
        for tile in well.get_tiles():
            assert tile.position.time == 0
            assert tile.position.channel in [0, 1, 2]
            assert tile.position.z == 0
            assert tile.position.y in [0]
            assert tile.position.x in [0, 512]
            assert tile.shape == (512, 512)


@pytest.fixture
def stack_acquisition(acquisition_dir):
    return StackAcquisition(acquisition_dir, alignment=TileAlignmentOptions.GRID)


def test_stack_acquistion(stack_acquisition: PlateAcquisition):
    wells = stack_acquisition.get_well_acquisitions()

    assert wells is not None
    assert len(wells) == 2
    # Full Stacks: 2 wells * 2 fields * 2 channels * 10 planes = 80 files
    # Single plane in stack: 2 wells * 2 fields * 1 channel * 1 plane = 4 files
    # Total of 84 files.
    # There are additionally 12 MIP files in the directory, but these are
    # ignored in this setup.
    assert len(wells[0]._files) + len(wells[0]._files) == 84

    channels = stack_acquisition.get_channel_metadata()
    assert len(channels) == 3
    ch = channels[0]
    assert ch.channel_index == 0
    assert ch.channel_name == "FITC_05"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == 536
    assert_almost_equal(ch.z_spacing, 5.0, decimal=4)

    ch = channels[1]
    assert ch.channel_index == 1
    assert ch.channel_name == "FITC_05"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == 536
    assert_almost_equal(ch.z_spacing, 5.0, decimal=4)

    ch = channels[3]
    assert ch.channel_index == 3
    assert ch.channel_name == "FITC_05"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == 536
    assert_almost_equal(ch.z_spacing, 5.0, decimal=4)

    for well in stack_acquisition.get_well_acquisitions():
        assert isinstance(well, WellAcquisition)
        assert len(well.get_tiles()) == 42
        for tile in well.get_tiles():
            assert tile.position.time == 0
            assert tile.position.channel in [0, 1, 3]
            assert tile.position.channel not in [4]
            assert tile.position.z in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            assert tile.position.y in [0]
            assert tile.position.x in [0, 512]
            assert tile.shape == (512, 512)


@pytest.fixture
def mixed_acquisition(acquisition_dir):
    return imagexpress.MixedAcquisition(
        acquisition_dir,
        alignment=TileAlignmentOptions.GRID,
    )


def test_mixed_acquisition(mixed_acquisition: PlateAcquisition):
    wells = mixed_acquisition.get_well_acquisitions()

    assert wells is not None
    assert len(wells) == 2
    # Stacks: 2 wells * 2 fields * 2 channels * 10 z-steps = 80 files
    # Single Plane: 2 wells * 2 fields * 1 channel = 4 files
    # MIP: 2 wells * 2 fields * 1 channel = 4 files
    # There are additionally 8 files for the MIPs of the stacks
    # (2 wells * 2 fields * 2 channels). But these are ignored.
    assert len(wells[0]._files) + len(wells[1]._files) == 80 + 4 + 4

    channels = mixed_acquisition.get_channel_metadata()
    assert len(channels) == 4
    ch = channels[0]
    assert ch.channel_index == 0
    assert ch.channel_name == "FITC_05"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == 536
    assert_almost_equal(ch.z_spacing, 5.0, decimal=4)

    ch = channels[1]
    assert ch.channel_index == 1
    assert ch.channel_name == "FITC_05"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == 536
    assert_almost_equal(ch.z_spacing, 5.0, decimal=4)

    ch = channels[2]
    assert ch.channel_index == 2
    assert ch.channel_name == "Maximum-Projection_FITC_05"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == 536
    assert_almost_equal(ch.z_spacing, 5.0, decimal=4)

    ch = channels[3]
    assert ch.channel_index == 3
    assert ch.channel_name == "FITC_05"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == 536
    assert_almost_equal(ch.z_spacing, 5.0, decimal=4)

    for well in mixed_acquisition.get_well_acquisitions():
        assert isinstance(well, WellAcquisition)
        assert len(well.get_tiles()) == 44
        for tile in well.get_tiles():
            assert tile.position.time == 0
            assert tile.position.channel in [0, 1, 2, 3]
            assert tile.position.z in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            assert tile.position.y in [0]
            assert tile.position.x in [0, 512]
            assert tile.shape == (512, 512)


@pytest.fixture
def dummy_plate():
    ImageXpressPlateAcquisition.__abstractmethods__ = set()

    @dataclass
    class Plate(ImageXpressPlateAcquisition):
        pass

    return Plate()


def test_raise_not_implemented_error(dummy_plate):
    with pytest.raises(NotImplementedError):
        dummy_plate._get_root_re()

    with pytest.raises(NotImplementedError):
        dummy_plate._get_filename_re()

    with pytest.raises(NotImplementedError):
        dummy_plate._get_z_spacing()


@pytest.fixture
def acquisition_dir_single_channel():
    return Path(__file__).parent.parent.parent.parent / "resources" / "SingleChannel"


@pytest.fixture
def single_channel_acquisition(acquisition_dir_single_channel):
    return SinglePlaneAcquisition(
        acquisition_dir_single_channel, alignment=TileAlignmentOptions.GRID
    )


def test_single_channel_acquistion(single_channel_acquisition: PlateAcquisition):
    wells = single_channel_acquisition.get_well_acquisitions()

    assert wells is not None
    assert len(wells) == 2
    # MIPs: 1 well has 2 fields * 1 channels = 2 files
    assert len(wells[0]._files) == 2
    assert len(wells[0]._files) == 2

    channels = single_channel_acquisition.get_channel_metadata()
    assert len(channels) == 1
    ch = channels[0]
    assert ch.channel_index == 0
    assert ch.channel_name == "Maximum-Projection_FITC_05"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == 536
    assert ch.z_spacing is None

    for well in single_channel_acquisition.get_well_acquisitions():
        assert isinstance(well, WellAcquisition)
        assert len(well.get_tiles()) == 2
        for tile in well.get_tiles():
            assert tile.position.time == 0
            assert tile.position.channel in [0]
            assert tile.position.z == 0
            assert tile.position.y in [0]
            assert tile.position.x in [0, 512]
            assert tile.shape == (512, 512)

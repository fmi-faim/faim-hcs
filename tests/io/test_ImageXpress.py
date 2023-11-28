from pathlib import Path

import pytest

from faim_hcs.io.acquisition import PlateAcquisition, WellAcquisition
from faim_hcs.io.ImageXpress import ImageXpressPlateAcquisition


@pytest.fixture
def acquisition_dir():
    return Path(__file__).parent.parent.parent / "resources" / "Projection-Mix"


@pytest.fixture
def acquisition(acquisition_dir):
    return ImageXpressPlateAcquisition(acquisition_dir)


def test_default(acquisition: PlateAcquisition):
    wells = acquisition._get_wells()

    assert wells is not None
    assert len(wells) == 2
    assert len(acquisition._files) == 96

    well_acquisitions = acquisition.well_acquisitions()

    channels = acquisition.channels()
    assert len(channels) == 4

    x_spacing = channels[0]["spatial-calibration-x"]
    y_spacing = channels[0]["spatial-calibration-y"]

    for well_acquisition in well_acquisitions:
        assert isinstance(well_acquisition, WellAcquisition)
        assert len(well_acquisition.files()) == 48
        files = well_acquisition.files()

        assert len(files[files["z"].isnull()]) == 2 * 3  # 2 fields, 3 channels (1,2,3)
        assert len(files[files["z"] == "1"]) == 2 * 3  # 2 fields, 3 channels (1,2,4)
        assert len(files[files["z"] == "10"]) == 2 * 2  # 2 fields, 2 channels (1,2)

        positions = well_acquisition.positions()

        assert positions is not None
        assert len(positions) == len(files)

        pixel_positions = well_acquisition.pixel_positions()

        assert pixel_positions.shape == (48, 2)
        assert pixel_positions[0, 1] == positions["pos_x"].iloc[0] / x_spacing
        assert pixel_positions[0, 0] == positions["pos_y"].iloc[0] / y_spacing

from os.path import exists, join
from pathlib import Path

import dask
import numpy as np
import pytest
import zarr
from distributed import Client, LocalCluster
from numcodecs import Blosc

from faim_hcs import dask_utils
from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.cellvoyager import StackAcquisition
from faim_hcs.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_hcs.hcs.plate import PlateLayout


def test_NGFFPlate():
    root_dir = "/path/to/root_dir"
    name = "plate_name"
    layout = PlateLayout.I18
    order_name = "order_name"
    barcode = "barcode"
    plate = NGFFPlate(
        root_dir=root_dir,
        name=name,
        layout=layout,
        order_name=order_name,
        barcode=barcode,
    )
    assert plate.root_dir == Path(root_dir)
    assert plate.name == name
    assert plate.layout == layout
    assert plate.order_name == order_name
    assert plate.barcode == barcode


@pytest.fixture
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("hcs_plate")


@pytest.fixture
def hcs_plate(tmp_dir):
    return NGFFPlate(
        root_dir=tmp_dir,
        name="plate_name",
        layout=PlateLayout.I96,
        order_name="order_name",
        barcode="barcode",
    )


@pytest.fixture
def plate_acquisition():
    return StackAcquisition(
        acquisition_dir=Path(__file__).parent.parent.parent
        / "resources"
        / "CV8000"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack",
        alignment=TileAlignmentOptions.GRID,
    )


def test__create_zarr_plate(tmp_dir, plate_acquisition, hcs_plate):
    converter = ConvertToNGFFPlate(hcs_plate)
    zarr_plate = converter._create_zarr_plate(plate_acquisition, wells=None)

    assert exists(join(tmp_dir, "plate_name.zarr"))
    assert zarr_plate.attrs["plate"]["name"] == "plate_name"
    assert zarr_plate.attrs["order_name"] == "order_name"
    assert zarr_plate.attrs["barcode"] == "barcode"
    assert zarr_plate.attrs["plate"]["field_count"] == 1
    assert zarr_plate.attrs["plate"]["wells"] == [
        {"columnIndex": 7, "path": "D/08", "rowIndex": 3},
        {"columnIndex": 2, "path": "E/03", "rowIndex": 4},
        {"columnIndex": 7, "path": "F/08", "rowIndex": 5},
    ]
    assert exists(join(tmp_dir, "plate_name.zarr", ".zgroup"))
    assert exists(join(tmp_dir, "plate_name.zarr", ".zattrs"))
    assert not exists(join(tmp_dir, "plate_name.zarr", "D"))
    assert not exists(join(tmp_dir, "plate_name.zarr", "E"))
    assert not exists(join(tmp_dir, "plate_name.zarr", "F"))

    zarr_plate_1 = converter._create_zarr_plate(plate_acquisition)
    assert zarr_plate_1 == zarr_plate


def test__out_chunks():
    out_chunks = ConvertToNGFFPlate._out_chunks(
        shape=(1, 2, 5, 10, 10),
        chunks=(1, 1, 5, 10, 5),
    )
    assert out_chunks == (1, 1, 5, 10, 5)

    out_chunks = ConvertToNGFFPlate._out_chunks(
        shape=(1, 2, 5, 10, 10),
        chunks=(5, 10, 10),
    )
    assert out_chunks == (1, 1, 5, 10, 10)


def test__get_storage_options():
    storage_options = ConvertToNGFFPlate._get_storage_options(
        storage_options=None,
        output_shape=(1, 2, 5, 10, 10),
        chunks=(5, 10, 5),
    )
    assert storage_options == {
        "dimension_separator": "/",
        "compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        "chunks": (1, 1, 5, 10, 5),
        "write_empty_chunks": False,
    }

    storage_options = ConvertToNGFFPlate._get_storage_options(
        storage_options={
            "dimension_separator": ".",
        },
        output_shape=(1, 2, 5, 10, 10),
        chunks=(5, 10, 5),
    )
    assert storage_options == {
        "dimension_separator": ".",
    }


def test__mean_cast_to():
    mean_cast_to = dask_utils.mean_cast_to(np.uint8)
    input = np.array([1.0, 2.0], dtype=np.float32)
    assert input.dtype == np.float32
    assert mean_cast_to(input).dtype == np.uint8
    assert mean_cast_to(input) == 1


def test__create_well_group(tmp_dir, plate_acquisition, hcs_plate):
    converter = ConvertToNGFFPlate(hcs_plate)
    zarr_plate = converter._create_zarr_plate(plate_acquisition)
    well_group = converter._create_well_group(
        plate=zarr_plate,
        well_acquisition=plate_acquisition.get_well_acquisitions()[0],
        well_sub_group="0",
    )
    assert exists(join(tmp_dir, "plate_name.zarr", "D", "08", "0"))
    assert isinstance(well_group, zarr.Group)


def test__stitch_well_image(tmp_dir, plate_acquisition, hcs_plate):
    converter = ConvertToNGFFPlate(hcs_plate)
    well_acquisition = plate_acquisition.get_well_acquisitions()[0]
    well_img_da = converter._stitch_well_image(
        chunks=(1, 1, 10, 1000, 1000),
        well_acquisition=well_acquisition,
        output_shape=plate_acquisition.get_common_well_shape(),
    )
    assert isinstance(well_img_da, dask.array.core.Array)
    assert well_img_da.shape == (1, 2, 4, 4000, 4000)
    assert well_img_da.dtype == np.uint16


def test__bin_yx(tmp_dir, plate_acquisition, hcs_plate):
    converter = ConvertToNGFFPlate(
        hcs_plate,
        yx_binning=2,
    )
    well_acquisition = plate_acquisition.get_well_acquisitions()[0]
    well_img_da = converter._stitch_well_image(
        chunks=(1, 1, 10, 1000, 1000),
        well_acquisition=well_acquisition,
        output_shape=plate_acquisition.get_common_well_shape(),
    )
    binned_yx = converter._bin_yx(well_img_da)
    assert isinstance(binned_yx, dask.array.core.Array)
    assert binned_yx.shape == (1, 2, 4, 2000, 2000)
    assert binned_yx.dtype == np.uint16

    converter._yx_binning = 1
    binned_yx = converter._bin_yx(well_img_da)
    assert isinstance(binned_yx, dask.array.core.Array)
    assert binned_yx.shape == (1, 2, 4, 4000, 4000)
    assert binned_yx.dtype == np.uint16


def test_run(tmp_dir, plate_acquisition, hcs_plate):
    converter = ConvertToNGFFPlate(
        hcs_plate,
        yx_binning=2,
        client=LocalCluster(
            n_workers=1, threads_per_worker=4, processes=False
        ).get_client(),
    )
    plate = converter.run(plate_acquisition, max_layer=2)
    plate.attrs["plate"]["wells"] == [
        {"columnIndex": 7, "path": "D/08", "rowIndex": 3},
        {"columnIndex": 2, "path": "E/03", "rowIndex": 4},
        {"columnIndex": 7, "path": "F/08", "rowIndex": 5},
    ]
    for well in ["D08", "E03", "F08"]:
        row, col = well[0], well[1:]
        path = join(tmp_dir, "plate_name.zarr", row, col, "0")
        assert exists(path)

        assert exists(join(path, "0"))
        assert exists(join(path, "1"))
        assert exists(join(path, ".zattrs"))
        assert exists(join(path, ".zgroup"))

        assert "acquisition_metadata" in plate[row][col]["0"].attrs.keys()
        assert "multiscales" in plate[row][col]["0"].attrs.keys()
        assert "omero" in plate[row][col]["0"].attrs.keys()

        assert exists(join(path, "0", ".zarray"))
        assert exists(join(path, "1", ".zarray"))

        assert plate[row][col]["0"]["0"].shape == (2, 4, 2000, 2000)
        assert plate[row][col]["0"]["1"].shape == (2, 4, 1000, 1000)


def test_run_selection(tmp_dir, plate_acquisition, hcs_plate):
    converter = ConvertToNGFFPlate(
        hcs_plate,
        yx_binning=2,
        client=LocalCluster(
            n_workers=1, threads_per_worker=4, processes=False
        ).get_client(),
    )
    plate = converter.run(plate_acquisition, max_layer=2, wells=["D08"])
    plate.attrs["plate"]["wells"] == [{"columnIndex": 7, "path": "D/08", "rowIndex": 3}]
    for well in ["D08"]:
        row, col = well[0], well[1:]
        path = join(tmp_dir, "plate_name.zarr", row, col, "0")
        assert exists(path)

        assert exists(join(path, "0"))
        assert exists(join(path, "1"))
        assert exists(join(path, ".zattrs"))
        assert exists(join(path, ".zgroup"))

        assert "acquisition_metadata" in plate[row][col]["0"].attrs.keys()
        assert "multiscales" in plate[row][col]["0"].attrs.keys()
        assert "omero" in plate[row][col]["0"].attrs.keys()

        assert exists(join(path, "0", ".zarray"))
        assert exists(join(path, "1", ".zarray"))

        assert plate[row][col]["0"]["0"].shape == (2, 4, 2000, 2000)
        assert plate[row][col]["0"]["1"].shape == (2, 4, 1000, 1000)


def test_provide_client(tmp_dir, plate_acquisition, hcs_plate):
    converter = ConvertToNGFFPlate(
        hcs_plate,
        yx_binning=2,
        client=Client(),
    )
    assert converter._client is not None
    assert converter._cluster_factory is None

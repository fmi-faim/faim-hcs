import json
from pathlib import Path

import pytest
from mobie.metadata.dataset_metadata import (
    create_dataset_metadata,
    create_dataset_structure,
)
from mobie.metadata.project_metadata import add_dataset, create_project_metadata
from mobie.validation import validate_project
from skimage.measure import label

from faim_hcs.io.MolecularDevicesImageXpress import parse_single_plane_multi_fields
from faim_hcs.MetaSeriesUtils import get_well_image_CYX
from faim_hcs.mobie import add_labels_view, add_wells_to_project
from faim_hcs.Zarr import (
    build_zarr_scaffold,
    write_cyx_image_to_well,
    write_labels_to_group,
)

ROOT_DIR = Path(__file__).parent.parent


@pytest.fixture
def files():
    return parse_single_plane_multi_fields(ROOT_DIR / "resources" / "Projection-Mix")


@pytest.fixture
def threshold():
    return 10000


@pytest.fixture
def zarr_group(files, zarr_dir, threshold):
    plate = build_zarr_scaffold(
        root_dir=zarr_dir,
        files=files,
        name="",
        layout=384,
        order_name="",
        barcode="",
    )
    for well in ["E07", "E08"]:
        img, hists, ch_metadata, metadata, roi_tables = get_well_image_CYX(
            well_files=files[files["well"] == well],
            channels=["w1"],
        )
        field = plate[well[0]][str(int(well[1:]))][0]
        write_cyx_image_to_well(img, hists, ch_metadata, metadata, field)
        labels = label(img > threshold)
        write_labels_to_group(
            labels=labels,
            labels_name="simple_threshold",
            parent_group=field,
        )
    return plate


@pytest.fixture
def mobie_project_folder(tmp_path_factory):
    return tmp_path_factory.mktemp("mobie")


@pytest.fixture
def zarr_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("zarr")


def test_full_mobie_project(zarr_group, mobie_project_folder):
    # setup OME-ZARR folder
    # setup MoBIE project
    dataset_name = "mobie_test_dataset"
    create_project_metadata(
        root=mobie_project_folder,
        #    description="MoBIE test project",
    )
    create_dataset_structure(
        root=mobie_project_folder,
        dataset_name=dataset_name,
        file_formats=["ome.zarr"],
    )
    create_dataset_metadata(
        dataset_folder=(mobie_project_folder / dataset_name),
        description="MoBIE test dataset",
        is2d=True,
    )
    add_dataset(
        root=mobie_project_folder,
        dataset_name=dataset_name,
        is_default=True,
    )

    # add_wells_to_project
    add_wells_to_project(
        plate=zarr_group,
        dataset_folder=(mobie_project_folder / dataset_name),
        well_group="0",
        view_name="default",
    )

    # validate 1
    validate_project(
        root=mobie_project_folder,
    )

    # add_labels_view
    with pytest.warns() as warning_record:
        add_labels_view(
            plate=zarr_group,
            dataset_folder=(mobie_project_folder / dataset_name),
            well_group="0",
            channel=0,
            label_name="simple_threshold",
            view_name="default",
            extra_properties=("area", "bbox"),
        )
    assert len(warning_record) == 1
    assert "A view with name default already exists for the dataset" in str(
        warning_record[0].message
    )

    # validate 2
    validate_project(
        root=mobie_project_folder,
    )

    dataset_json_path: Path = mobie_project_folder / dataset_name / "dataset.json"

    assert dataset_json_path.exists()

    with open(dataset_json_path) as f:
        json_data = json.load(f)
    assert "views" in json_data
    assert "sourceDisplays" in json_data["views"]["default"]
    assert "sourceTransforms" in json_data["views"]["default"]

    # wells table
    table_path: Path = (
        mobie_project_folder / dataset_name / "tables" / "wells" / "default.tsv"
    )
    assert table_path.exists()

    # segmentation table for E07
    label_table_path: Path = (
        mobie_project_folder
        / dataset_name
        / "tables"
        / "E07_simple_threshold"
        / "default.tsv"
    )
    assert label_table_path.exists()

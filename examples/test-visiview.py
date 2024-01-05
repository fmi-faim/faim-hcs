import os
import re
import shutil
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from numpy._typing import NDArray
from tifffile import TiffFile, imread

from faim_hcs.hcs.acquisition import (
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_hcs.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_hcs.hcs.plate import PlateLayout
from faim_hcs.io.ChannelMetadata import ChannelMetadata
from faim_hcs.stitching import Tile, stitching_utils
from faim_hcs.stitching.Tile import TilePosition
from faim_hcs.utils import rgb_to_hex, wavelength_to_rgb


class JohnsonTile(Tile):
    def load_data(self) -> NDArray:
        with TiffFile(self.path) as tif:
            return tif.asarray(self.position.z)


class JohnsonWellAcquisition(WellAcquisition):
    def get_axes(self) -> list[str]:
        return ["c", "z", "y", "x"]

    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        z_spacing: Optional[float],
        yx_spacing: tuple[float, float],
    ) -> None:
        self._z_spacing = z_spacing
        self._yx_spacing = yx_spacing
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=None,
            illumination_correction_matrices=None,
        )

    def _assemble_tiles(self) -> list[Tile]:
        tiles = []
        for i, row in self._files.iterrows():
            file = row["path"]
            time_point = 0
            channel = row["channel"]
            y = int(row["Y"] / self._yx_spacing[0])
            x = int(row["X"] / self._yx_spacing[1])

            shape = imread(file).shape
            for z in range(shape[0]):
                tiles.append(
                    JohnsonTile(
                        path=file,
                        shape=shape[1:],
                        position=TilePosition(
                            time=time_point,
                            channel=int(channel[1:]),
                            z=z,
                            y=y,
                            x=x,
                        ),
                    )
                )
        return tiles

    def get_z_spacing(self) -> Optional[float]:
        return self._z_spacing

    def get_yx_spacing(self) -> tuple[float, float]:
        return self._yx_spacing


class JohnsonPlateAcquisition(PlateAcquisition):
    def __init__(
        self,
        acquisition_dir: Union[Path, str],
    ):
        self._z_spacing = 5.0
        self._spacing = (0.534036458, 0.534036458)
        super().__init__(
            acquisition_dir=acquisition_dir,
            alignment=TileAlignmentOptions.STAGE_POSITION,
            background_correction_matrices=None,
            illumination_correction_matrices=None,
        )

    def _parse_files(self) -> pd.DataFrame:
        filename_re = re.compile(
            r"(?P<project_number>\d+)-(?P<project_iteration>\d+)-("
            r"?P<slide_number>Sl\d+)-(?P<unknown>[A-Z]{1}[0-9]{1,2}-[A-Z]{1}[0-9]{1,2})-(?P<magnification>\d+x)-_(?P<region>Region-\d+)_(?P<n_tiles>nTiles-\d+)_1_(?P<channel>w\d+)conf(?P<wavelength>\d+)Virtex_(?P<site>s\d+).ome.tif"
        )
        files = []
        for root, _, filenames in os.walk(self._acquisition_dir):
            for f in filenames:
                m_filename = filename_re.fullmatch(f)
                if m_filename:
                    row = m_filename.groupdict()
                    row["path"] = str(Path(root) / f)
                    files.append(row)

        files = pd.DataFrame(files)

        def create_well_name(row):
            if row["region"].split("-")[1] in ["1", "2", "3", "4", "5", "6", "7"]:
                return f"A{str(int(row['region'].split('-')[1])).zfill(2)}"
            elif row["region"].split("-")[1] in [
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
            ]:
                return f"B{str(int(row['region'].split('-')[1]) - 7).zfill(2)}"

        files["well"] = files.apply(create_well_name, axis=1)

        positions = []
        for region in files["region"].unique():
            region_files = files[files["region"] == region]

            row = region_files.iloc[0]
            stage_file = (
                row["project_number"]
                + "-"
                + row["project_iteration"]
                + "-"
                + row["slide_number"]
                + "-"
                + row["unknown"]
                + "-"
                + row["magnification"]
                + "-_"
                + row["region"]
                + "_"
                + row["n_tiles"]
                + "_.stg"
            )

            with open(join(self._acquisition_dir, stage_file)) as f:
                for i in range(4):
                    f.readline()

                for i, line in enumerate(f.readlines()):
                    line = line.split(",")
                    tmp = {}
                    tmp["region"] = row["region"]
                    tmp["site"] = f"s{i+1}"
                    tmp["X"] = float(line[1])
                    tmp["Y"] = float(line[2])

                    positions.append(tmp)

        positions = pd.DataFrame(positions)

        return files.join(
            positions.set_index(["region", "site"]), on=["region", "site"]
        )

    def get_channel_metadata(self) -> dict[int, ChannelMetadata]:
        _files = self._wells[0]._files
        channels = {}
        exposure_times = [300, 200, 200, 40]
        for i, ch in enumerate(_files["channel"].unique()):
            row = _files[_files["channel"] == ch].iloc[0]
            channels[i] = ChannelMetadata(
                channel_index=i,
                channel_name=row["channel"] + "_Conf" + row["wavelength"],
                display_color=rgb_to_hex(*wavelength_to_rgb(int(row["wavelength"]))),
                spatial_calibration_x=self._spacing[1],
                spatial_calibration_y=self._spacing[0],
                spatial_calibration_units="um",
                z_spacing=self._z_spacing,
                wavelength=int(row["wavelength"]),
                exposure_time=exposure_times[i],
                exposure_time_unit="ms",
                objective=row["magnification"],
            )

        return channels

    def _build_well_acquisitions(self, files: pd.DataFrame) -> list["WellAcquisition"]:
        wells = []
        for well in files["well"].unique():
            wells.append(
                JohnsonWellAcquisition(
                    files=files[files["well"] == well],
                    alignment=self._alignment,
                    z_spacing=self._z_spacing,
                    yx_spacing=self._spacing,
                )
            )
        return wells


def main():
    # lc = LocalClusterFactory(
    #     n_workers=12,
    #     threads_per_worker=2,
    #     memory_limit="16GB",
    # )
    from distributed import LocalCluster

    lc = LocalCluster(
        n_workers=10,
        threads_per_worker=2,
        processes=False,
        memory_limit="16GB",
    )
    client = lc.get_client()  # noqa
    plate = JohnsonPlateAcquisition(acquisition_dir="/home/tibuch/Data/matt-test")
    name = "2304-11-Sl1-A1-E2-20x"
    shutil.rmtree(name + ".zarr", ignore_errors=True)

    converter = ConvertToNGFFPlate(
        ngff_plate=NGFFPlate(
            root_dir=".",
            name=name,
            layout=PlateLayout.I96,
            order_name="order",
            barcode="barcode",
        ),
        yx_binning=1,
        fuse_func=stitching_utils.fuse_linear,
        client=client,
    )
    converter.run(
        wells=["A01"],  # , "A02", "A03", "A04", "A05"],
        plate_acquisition=plate,
        well_sub_group="0",
        chunks=(11, 1024, 1024),
        max_layer=3,
    )


if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print(end - start)

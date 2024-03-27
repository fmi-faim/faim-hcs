from pathlib import Path
from typing import Optional, Union

import pandas as pd
from numpy._typing import NDArray
from tifffile import TiffFile

from faim_hcs.hcs.acquisition import TileAlignmentOptions, WellAcquisition
from faim_hcs.io.ChannelMetadata import ChannelMetadata
from faim_hcs.stitching import Tile
from faim_hcs.stitching.Tile import TilePosition
from faim_hcs.visiview.ome_companion_utils import parse_basic_metadata
from faim_hcs.visiview.StackedTile import StackedTile


class RegionAcquisitionSTK(WellAcquisition):
    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, NDArray]],
        illumination_correction_matrices: Optional[dict[str, NDArray]],
        axes: list[str] = ["c", "z", "y", "x"],
        *,
        memmap: bool = True,
    ):
        path = files.iloc[0]["path"]
        with TiffFile(path) as tif:
            metadata = tif.stk_metadata
            if metadata is None:
                raise ValueError(f"STK metadata is missing. Please check " f"{path}")
            x_spacing = metadata["XCalibration"]
            y_spacing = metadata["YCalibration"]
            self._yx_spacing = (y_spacing, x_spacing)
            self._z_spacing = metadata["ZDistance"].mean()
            self.tile_shape = tif.asarray().shape

        self._axes = axes
        self._memmap = memmap
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _assemble_tiles(self) -> list[Tile]:
        tiles = []
        for i, row in self._files.iterrows():
            file = row["path"]
            time_point = row["time"]
            channel = row["channel"]

            tiles.append(
                StackedTile(
                    path=file,
                    shape=self.tile_shape,
                    position=TilePosition(
                        time=time_point,
                        channel=channel,
                        z=0,
                        y=int(row["Y"] / self._yx_spacing[0]),
                        x=int(row["X"] / self._yx_spacing[1]),
                    ),
                    memmap=self._memmap,
                )
            )

        return tiles

    def get_z_spacing(self) -> Optional[float]:
        return self._z_spacing

    def get_yx_spacing(self) -> tuple[float, float]:
        return self._yx_spacing

    def get_axes(self) -> list[str]:
        return self._axes


class RegionAcquisitionOME(WellAcquisition):
    def __init__(
        self,
        files: pd.DataFrame,
        ome_xml: Union[Path, str],
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, NDArray]],
        illumination_correction_matrices: Optional[dict[str, NDArray]],
        axes: list[str] = ["c", "z", "y", "x"],
        memmap: bool = True,
    ):
        self.metadata = parse_basic_metadata(companion_file=ome_xml)
        self.stage_positions = self.metadata["stage_positions"]
        path = files.iloc[0]["path"]
        with TiffFile(path) as tif:
            self.tile_shape = tif.asarray().shape

        self._axes = axes
        self._memmap = memmap
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _assemble_tiles(self) -> list[Tile]:
        tiles = []
        for i, row in self._files.iterrows():
            file = row["path"]
            time_point = row["time"]
            channel = row["channel"]

            tiles.append(
                StackedTile(
                    path=file,
                    shape=self.tile_shape,
                    position=TilePosition(
                        time=time_point,
                        channel=channel,
                        z=0,
                        y=int(
                            self.stage_positions[row["well"]][0]
                            / self.metadata["yx_spacing"][0]
                        ),
                        x=int(
                            self.stage_positions[row["well"]][1]
                            / self.metadata["yx_spacing"][1]
                        ),
                    ),
                    memmap=self._memmap,
                )
            )

        return tiles

    def get_z_spacing(self) -> Optional[float]:
        return self.metadata["z_spacing"]

    def get_yx_spacing(self) -> tuple[float, float]:
        return self.metadata["yx_spacing"]

    def get_channels(self) -> list[ChannelMetadata]:
        return self.metadata["channels"]

    def get_axes(self) -> list[str]:
        return self._axes

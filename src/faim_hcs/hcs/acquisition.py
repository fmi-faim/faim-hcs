from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from faim_hcs.io.ChannelMetadata import ChannelMetadata
from faim_hcs.stitching import Tile


class TileAlignmentOptions(Enum):
    """Tile alignment options."""

    STAGE_POSITION = "StageAlignment"
    GRID = "GridAlignment"


class PlateAcquisition(ABC):
    _acquisition_dir = None
    _files = None
    _alignment: TileAlignmentOptions = None
    _background_correction_matrices: Optional[dict[str, Union[Path, str]]]
    _illumination_correction_matrices: Optional[dict[str, Union[Path, str]]]

    def __init__(
        self,
        acquisition_dir: Union[Path, str],
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, Union[Path, str]]],
        illumination_correction_matrices: Optional[dict[str, Union[Path, str]]],
    ) -> None:
        self._acquisition_dir = acquisition_dir
        self._files = self._parse_files()
        self._alignment = alignment
        self._background_correction_matrices = background_correction_matrices
        self._illumination_correction_matrices = illumination_correction_matrices
        super().__init__()

    @abstractmethod
    def _parse_files(self) -> pd.DataFrame:
        """Parse all files in the acquisition directory.

        Returns
        -------
        DataFrame
            Table of all files in the acquisition.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_well_acquisitions(self) -> list["WellAcquisition"]:
        """List of wells."""
        raise NotImplementedError()

    @abstractmethod
    def get_channel_metadata(self) -> dict[int, ChannelMetadata]:
        """Channel metadata."""
        raise NotImplementedError()

    def get_well_names(self) -> Iterable[str]:
        for well in self.get_well_acquisitions():
            yield well.name

    def get_omero_channel_metadata(self) -> list[dict]:
        ome_channels = []
        ch_metadata = self.get_channel_metadata()
        max_channel = max(list(ch_metadata.keys()))
        for index in range(max_channel + 1):
            if index in ch_metadata.keys():
                metadata = ch_metadata[index]
                ome_channels.append(
                    {
                        "active": True,
                        "coefficient": 1,
                        "color": metadata.display_color,
                        "family": "linear",
                        "inverted": False,
                        "label": metadata.channel_name,
                        "wavelength_id": f"C{str(metadata.channel_index + 1).zfill(2)}",
                        "window": {
                            "min": np.iinfo(np.uint16).min,
                            "max": np.iinfo(np.uint16).max,
                            "start": np.iinfo(np.uint16).min,
                            "end": np.iinfo(np.uint16).max,
                        },
                    }
                )
            elif index < max_channel:
                ome_channels.append(
                    {
                        "active": False,
                        "coefficient": 1,
                        "color": "#000000",
                        "family": "linear",
                        "inverted": False,
                        "label": "empty",
                        "wavelength_id": f"C{str(index + 1).zfill(2)}",
                        "window": {
                            "min": np.iinfo(np.uint16).min,
                            "max": np.iinfo(np.uint16).max,
                            "start": np.iinfo(np.uint16).min,
                            "end": np.iinfo(np.uint16).max,
                        },
                    }
                )

        return ome_channels

    def get_common_well_shape(self) -> tuple[int, int, int, int, int]:
        """
        Compute the maximum well extent such that each well is covered.

        Returns
        -------
        tuple[int, int, int, int, int]
            (time, channel, z, y, x)
        """
        well_shapes = []
        for well in self.get_well_acquisitions():
            well_shapes.append(well.get_shape())

        return tuple(np.max(well_shapes, axis=0))


class WellAcquisition(ABC):
    name: str = None
    _files = None
    _alignment: TileAlignmentOptions = None
    _background_correction_matrices: Optional[dict[str, Union[Path, str]]]
    _illumincation_correction_matrices: Optional[dict[str, Union[Path, str]]]
    _tiles = None

    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, Union[Path, str]]],
        illumination_correction_matrices: Optional[dict[str, Union[Path, str]]],
    ) -> None:
        assert (
            files["well"].nunique() == 1
        ), "WellAcquisition must contain files from a single well."
        self.name = files["well"].iloc[0]
        self._files = files
        self._alignment = alignment
        self._background_correction_matrices = background_correction_matrices
        self._illumincation_correction_matrices = illumination_correction_matrices
        self._tiles = self._align_tiles(tiles=self._assemble_tiles())
        super().__init__()

    @abstractmethod
    def _assemble_tiles(self) -> list[Tile]:
        """Parse all tiles in the well."""
        raise NotImplementedError()

    def get_dtype(self) -> np.dtype:
        return self._tiles[0].load_data().dtype

    def _align_tiles(self, tiles: list[Tile]) -> list[Tile]:
        if self._alignment == TileAlignmentOptions.STAGE_POSITION:
            from faim_hcs.alignment import StageAlignment

            return StageAlignment(tiles=tiles).get_tiles()

        if self._alignment == TileAlignmentOptions.GRID:
            from faim_hcs.alignment import GridAlignment

            return GridAlignment(tiles=tiles).get_tiles()

        raise ValueError(f"Unknown alignment option: {self._alignment}")

    def get_tiles(self) -> list[Tile]:
        """List of tiles."""
        return self._tiles

    def get_row_col(self) -> tuple[str, str]:
        return self.name[0], self.name[1:]

    @abstractmethod
    def get_axes(self) -> list[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_yx_spacing(self) -> tuple[float, float]:
        raise NotImplementedError()

    @abstractmethod
    def get_z_spacing(self) -> Optional[float]:
        raise NotImplementedError()

    def get_coordinate_transformations(
        self, max_layer: int, yx_binning: int
    ) -> list[dict[str, Any]]:
        transformations = []
        for s in range(max_layer + 1):
            if self.get_z_spacing() is not None:
                transformations.append(
                    [
                        {
                            "scale": [
                                1.0,
                                self.get_z_spacing(),
                                self.get_yx_spacing()[0] * yx_binning * 2**s,
                                self.get_yx_spacing()[1] * yx_binning * 2**s,
                            ],
                            "type": "scale",
                        }
                    ]
                )
            else:
                transformations.append(
                    [
                        {
                            "scale": [
                                1.0,
                                self.get_yx_spacing()[0] * yx_binning * 2**s,
                                self.get_yx_spacing()[1] * yx_binning * 2**s,
                            ],
                            "type": "scale",
                        }
                    ]
                )

        return transformations

    def get_shape(self):
        """
        Compute the theoretical shape of the stitched well image.
        """
        tile_extents = []
        for tile in self._tiles:
            tile_extents.append(tile.get_position() + np.array((1, 1, 1) + tile.shape))
        return tuple(np.max(tile_extents, axis=0))
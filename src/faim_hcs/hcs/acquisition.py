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
    _wells = None
    _alignment: TileAlignmentOptions = None
    _background_correction_matrices: Optional[dict[str, Union[Path, str]]]
    _illumination_correction_matrices: Optional[dict[str, Union[Path, str]]]
    _common_well_shape: tuple[int, int, int, int, int] = None

    def __init__(
        self,
        acquisition_dir: Union[Path, str],
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, Union[Path, str]]],
        illumination_correction_matrices: Optional[dict[str, Union[Path, str]]],
    ) -> None:
        self._acquisition_dir = acquisition_dir
        self._alignment = alignment
        self._background_correction_matrices = background_correction_matrices
        self._illumination_correction_matrices = illumination_correction_matrices
        self._wells = self._build_well_acquisitions(self._parse_files())
        super().__init__()

    @abstractmethod
    def _parse_files(self) -> pd.DataFrame:
        """Parse all files in the acquisition directory.

        Returns
        -------
        DataFrame
            Table of all files in the acquisition.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_well_acquisitions(self, files: pd.DataFrame) -> list["WellAcquisition"]:
        """List of wells."""
        raise NotImplementedError

    def get_well_acquisitions(
        self, selection: Optional[list[str]] = None
    ) -> list["WellAcquisition"]:
        if selection is None:
            return self._wells
        else:
            return [well for well in self._wells if well.name in selection]

    @abstractmethod
    def get_channel_metadata(self) -> dict[int, ChannelMetadata]:
        """Channel metadata."""
        raise NotImplementedError

    def get_well_names(self, wells: Optional[list[str]] = None) -> Iterable[str]:
        """
        Get the names of all wells in the acquisition.
        """
        for well in self.get_well_acquisitions(selection=wells):
            yield well.name

    def get_omero_channel_metadata(self) -> list[dict]:
        """
        Get the channel metadata in OMERO format.

        Returns
        -------
            List of channel metadata.
        """
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
            else:
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
            (time, channel, z, y, x)
        """
        if self._common_well_shape is None:
            well_shapes = []
            for well in self.get_well_acquisitions():
                well_shapes.append(well.get_shape())

            self._common_well_shape = tuple(np.max(well_shapes, axis=0))

        return self._common_well_shape


class WellAcquisition(ABC):
    """
    A single well of a plate acquisition.
    """

    name: str = None
    _files = None
    _alignment: TileAlignmentOptions = None
    _background_correction_matrices: Optional[dict[str, Union[Path, str]]]
    _illumination_correction_matrices: Optional[dict[str, Union[Path, str]]]
    _tiles = None
    _shape: tuple[int, int] = None
    _dtype: np.dtype = None

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
        self._illumination_correction_matrices = illumination_correction_matrices
        self._tiles = self._align_tiles(tiles=self._assemble_tiles())
        super().__init__()

    @abstractmethod
    def _assemble_tiles(self) -> list[Tile]:
        """Parse all tiles in the well."""
        raise NotImplementedError

    def get_dtype(self) -> np.dtype:
        """
        Get the data type of the well acquisition.

        Returns
        -------
            type
        """
        if self._dtype is None:
            self._dtype = self._tiles[0].load_data().dtype

        return self._dtype

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
        """
        Get the row and column of the well acquisition.

        Returns
        -------
            row, column
        """
        return self.name[0], self.name[1:]

    @abstractmethod
    def get_axes(self) -> list[str]:
        """
        Get the axes of the well acquisition.
        """
        raise NotImplementedError

    @abstractmethod
    def get_yx_spacing(self) -> tuple[float, float]:
        """
        Get the yx spacing of the well acquisition.
        """
        raise NotImplementedError

    @abstractmethod
    def get_z_spacing(self) -> Optional[float]:
        """
        Get the z spacing of the well acquisition.
        """
        raise NotImplementedError

    def get_coordinate_transformations(
        self,
        max_layer: int,
        yx_binning: int,
        ndim: int,
    ) -> list[dict[str, Any]]:
        """
        Get the NGFF conform coordinate transformations for the well
        acquisition.

        Parameters
        ----------
        max_layer : Maximum layer of the resolution pyramid.
        yx_binning : Bin factor of the yx resolution.
        ndim : Number of dimensions of the data.

        Returns
        -------
            List of coordinate transformations.
        """
        transformations = []
        for s in range(max_layer + 1):
            if self.get_z_spacing() is not None:
                transformations.append(
                    [
                        {
                            "scale": [
                                1.0,
                            ]
                            * (ndim - 3)
                            + [
                                float(self.get_z_spacing()),
                                float(self.get_yx_spacing()[0] * yx_binning * 2**s),
                                float(self.get_yx_spacing()[1] * yx_binning * 2**s),
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
                            ]
                            * (ndim - 2)
                            + [
                                float(self.get_yx_spacing()[0] * yx_binning * 2**s),
                                float(self.get_yx_spacing()[1] * yx_binning * 2**s),
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
        if self._shape is None:
            tile_extents = []
            for tile in self._tiles:
                tile_extents.append(
                    tile.get_position()
                    + np.array((1,) * (5 - len(tile.shape)) + tile.shape)
                )
            self._shape = tuple(np.max(tile_extents, axis=0))

        return self._shape

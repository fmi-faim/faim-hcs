import re
from decimal import Decimal
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy._typing import NDArray

from faim_hcs.io.acquisition import (
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_hcs.io.ImageXpress import ImageXpressWellAcquisition
from faim_hcs.io.metadata import ChannelMetadata
from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff_metadata


class StackAcquisition(PlateAcquisition):
    def __init__(
        self,
        acquisition_dir: Union[Path, str],
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, NDArray]] = None,
        illumination_correction_matrices: Optional[NDArray] = None,
    ):
        super().__init__(acquisition_dir=acquisition_dir, alignment=alignment)
        self._z_spacing = self._compute_z_spacing()
        self._background_correction_matriecs = background_correction_matrices
        self._illumination_correction_matrices = illumination_correction_matrices

    def _get_root_re(self) -> re.Pattern:
        return re.compile(
            r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)(?:[\/\\]ZStep_(?P<z>\d+))"
        )

    def _get_filename_re(self) -> re.Pattern:
        return re.compile(
            r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_(?P<field>s\d+)_(?P<channel>w[1-9]{1})(?!_thumb)(?P<md_id>.*)(?P<ext>.tif)"
        )

    def get_well_acquisitions(self) -> list[WellAcquisition]:
        return [
            ImageXpressWellAcquisition(
                files=self._files[self._files["well"] == well],
                alignment=self._alignment,
                z_spacing=self._z_spacing,
                background_correction_matrices=self._background_correction_matriecs,
                illumination_correction_matrices=self._illumination_correction_matrices,
            )
            for well in self._files["well"].unique()
        ]

    def get_channel_metadata(self) -> dict[str, ChannelMetadata]:
        ch_metadata = {}
        for ch in self._files["channel"].unique():
            channel_files = self._files[self._files["channel"] == ch]
            path = channel_files["path"].iloc[0]
            metadata = load_metaseries_tiff_metadata(path=path)
            from faim_hcs.MetaSeriesUtils import _build_ch_metadata

            channel_metadata = _build_ch_metadata(metadata)
            ch_metadata[ch] = ChannelMetadata(
                channel_index=int(ch[1:]) - 1,
                channel_name=ch,
                display_color=channel_metadata["display-color"],
                spatial_calibration_x=metadata["spatial-calibration-x"],
                spatial_calibration_y=metadata["spatial-calibration-y"],
                spatial_calibration_units=metadata["spatial-calibration-units"],
                z_spacing=self._z_spacing,
                wavelength=channel_metadata["wavelength"],
                exposure_time=channel_metadata["exposure-time"],
                exposure_time_unit=channel_metadata["exposure-time-unit"],
                objective=metadata["_MagSetting_"],
            )

        return ch_metadata

    def _compute_z_spacing(
        self,
    ) -> Optional[float]:
        if "z" in self._files.columns:
            channels_with_stack = self._files[self._files["z"] == "2"][
                "channel"
            ].unique()
        else:
            return None

        plane_positions = {}

        for i, row in self._files[
            self._files["channel"].isin(channels_with_stack)
        ].iterrows():
            file = row["path"]
            if "z" in row.keys() and row["z"] is not None:
                z = int(row["z"])
                metadata = load_metaseries_tiff_metadata(file)
                z_position = metadata["stage-position-z"]
                if z in plane_positions.keys():
                    plane_positions[z].append(z_position)
                else:
                    plane_positions[z] = [z_position]

        if len(plane_positions) > 1:
            plane_positions = dict(sorted(plane_positions.items()))
            average_z_positions = []
            for z, positions in plane_positions.items():
                average_z_positions.append(np.mean(positions))

            precision = -Decimal(str(plane_positions[1][0])).as_tuple().exponent
            z_step = np.round(np.mean(np.diff(average_z_positions)), decimals=precision)
            return z_step
        else:
            return None
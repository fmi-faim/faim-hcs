import os
import re
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm

from faim_hcs.hcs.acquisition import (
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_hcs.hcs.imagexpress.ImageXpressWellAcquisition import (
    ImageXpressWellAcquisition,
)
from faim_hcs.io.ChannelMetadata import ChannelMetadata
from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff_metadata
from faim_hcs.utils import rgb_to_hex, wavelength_to_rgb


class ImageXpressPlateAcquisition(PlateAcquisition):
    def __init__(
        self,
        acquisition_dir: Union[Path, str],
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, Union[Path, str]]] = None,
        illumination_correction_matrices: Optional[dict[str, Union[Path, str]]] = None,
    ):
        super().__init__(
            acquisition_dir=acquisition_dir,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _parse_files(self) -> pd.DataFrame:
        """Parse all files in the acquisition directory.

        Returns
        -------
        DataFrame
            Table of all files in the acquisition.
        """
        return pd.DataFrame(
            ImageXpressPlateAcquisition._list_and_match_files(
                root_dir=self._acquisition_dir,
                root_re=self._get_root_re(),
                filename_re=self._get_filename_re(),
            )
        )

    @staticmethod
    def _list_and_match_files(
        root_dir: Union[Path, str],
        root_re: re.Pattern,
        filename_re: re.Pattern,
    ) -> list[list[dict[dict, str]]]:
        files = []
        for root, _, filenames in os.walk(root_dir):
            m_root = root_re.fullmatch(root)
            if m_root:
                for f in filenames:
                    m_filename = filename_re.fullmatch(f)
                    if m_filename:
                        row = m_root.groupdict()
                        row |= m_filename.groupdict()
                        if "channel" not in row or row["channel"] is None:
                            row["channel"] = "w1"
                        row["path"] = str(Path(root).joinpath(f))
                        files.append(row)
        return files

    @abstractmethod
    def _get_root_re(self) -> re.Pattern:
        """Regular expression for matching the root directory of the acquisition."""
        raise NotImplementedError

    @abstractmethod
    def _get_filename_re(self) -> re.Pattern:
        """Regular expression for matching the filename of the acquisition."""
        raise NotImplementedError

    def _build_well_acquisitions(self, files: pd.DataFrame) -> list[WellAcquisition]:
        wells = []
        for well in tqdm(files["well"].unique()):
            wells.append(
                ImageXpressWellAcquisition(
                    files=files[files["well"] == well],
                    alignment=self._alignment,
                    z_spacing=self._get_z_spacing(),
                    background_correction_matrices=self._background_correction_matrices,
                    illumination_correction_matrices=self._illumination_correction_matrices,
                )
            )

        return wells

    @abstractmethod
    def _get_z_spacing(self) -> Optional[float]:
        raise NotImplementedError

    def get_channel_metadata(self) -> dict[int, ChannelMetadata]:
        ch_metadata = {}
        _files = self._wells[0]._files
        for ch in _files["channel"].unique():
            channel_files = _files[_files["channel"] == ch]
            path = channel_files["path"].iloc[0]
            metadata = load_metaseries_tiff_metadata(path=path)
            index = int(ch[1:]) - 1
            if "Z Projection Method" in metadata.keys():
                name = (
                    f"{metadata['Z Projection Method'].replace(' ', '-')}-Projection_"
                    f"{metadata['_IllumSetting_']}"
                )
            else:
                name = metadata["_IllumSetting_"]
            ch_metadata[index] = ChannelMetadata(
                channel_index=index,
                channel_name=name,
                display_color=rgb_to_hex(*wavelength_to_rgb(metadata["wavelength"])),
                spatial_calibration_x=metadata["spatial-calibration-x"],
                spatial_calibration_y=metadata["spatial-calibration-y"],
                spatial_calibration_units=metadata["spatial-calibration-units"],
                z_spacing=self._get_z_spacing(),
                wavelength=metadata["wavelength"],
                exposure_time=float(metadata["Exposure Time"].split(" ")[0]),
                exposure_time_unit=metadata["Exposure Time"].split(" ")[1],
                objective=metadata["_MagSetting_"],
            )

        assert min(ch_metadata.keys()) == 0, "Channel indices must start at 0."
        return ch_metadata

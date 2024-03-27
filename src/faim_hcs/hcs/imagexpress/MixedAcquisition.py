import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from numpy._typing import NDArray

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.imagexpress import StackAcquisition


class MixedAcquisition(StackAcquisition):
    """Image stack acquisition with Projectsion acquired with a Molecular
    Devices ImageXpress Micro Confocal system.

    MIP-2P-2sub-Stack --> {name} [Optional]
    └── 2023-02-21 --> {date}
        └── 1334 --> {acquisition id}
            ├── Projection-Mix_E07_s1_w1E94C24BD-45E4-450A-9919-257C714278F7.tif
            ├── Projection-Mix_E07_s1_w1_thumb4BFD4018-E675-475E-B5AB-2E959E6B6DA1.tif
            ├── ...
            ├── Projection-Mix_E08_s2_w3CCE83D85-0912-429E-9F18-716A085BB5BC.tif
            ├── Projection-Mix_E08_s2_w3_thumb4D88636E-181E-4AF6-BC53-E7A435959C8F.tif
            ├── ZStep_1
            │   ├── Projection-Mix_E07_s1_w1E78EB128-BD0D-4D94-A6AD-3FF28BB1B105.tif
            │   ├── Projection-Mix_E07_s1_w1_thumb187DE64B-038A-4671-BF6B-683721723769.tif
            │   ├── Projection-Mix_E07_s1_w2C0A49256-E289-4C0F-ADC9-F7728ABDB141.tif
            │   ├── Projection-Mix_E07_s1_w2_thumb57D4B151-71BF-480E-8CC4-C23A2690B763.tif
            │   ├── Projection-Mix_E07_s1_w427CCB2E4-1BF4-45E7-8BC7-264B48EF9C4A.tif
            │   ├── Projection-Mix_E07_s1_w4_thumb555647D0-77F1-4A43-9472-AE509F95E236.tif
            │   ├── ...
            │   └── Projection-Mix_E08_s2_w4_thumbD2785594-4F49-464F-9F80-1B82E30A560A.tif
            ├── ...
            └── ZStep_9
                ├── Projection-Mix_E07_s1_w1091EB8A5-272A-466D-B8A0-7547C6BA392B.tif
                ├── ...
                └── Projection-Mix_E08_s2_w2_thumb210C0D5D-C20E-484D-AFB2-EFE669A56B84.tif

    Image data is stored in {name}_{well}_{field}_w{channel}{md_id}.tif.
    The *_thumb*.tif files, used by Molecular Devices as preview, are ignored.
    """

    def __init__(
        self,
        acquisition_dir: Union[Path, str],
        alignment: TileAlignmentOptions,
        background_correction_matrix: Optional[dict[str, NDArray]] = None,
        illumination_correction_matrix: Optional[NDArray] = None,
    ):
        super().__init__(
            acquisition_dir=acquisition_dir,
            alignment=alignment,
            background_correction_matrices=background_correction_matrix,
            illumination_correction_matrices=illumination_correction_matrix,
        )

    def _parse_files(self) -> pd.DataFrame:
        files = self._filter_mips(super()._parse_files())
        self._z_spacing = self._compute_z_spacing(files)
        return files

    def _filter_mips(self, files: pd.DataFrame) -> pd.DataFrame:
        """Remove MIP files if the whole stack was acquired."""
        _files = files.copy()
        for ch in _files["channel"].unique():
            channel_files = _files[_files["channel"] == ch]
            z_positions = channel_files["z"].unique()
            has_mip = None in z_positions
            has_stack = len(z_positions) > 1
            if has_mip and has_stack:
                _files.drop(
                    _files[(_files["channel"] == ch) & (_files["z"].isna())].index,
                    inplace=True,
                )
        return _files

    def _get_root_re(self) -> re.Pattern:
        return re.compile(
            r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)(?:[\/\\]ZStep_(?P<z>\d+))?.*"
        )

    def _get_filename_re(self) -> re.Pattern:
        return re.compile(
            r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_?(?P<field>s\d+)?_?(?P<channel>w[1-9]{1})?(?!_thumb)(?P<md_id>[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12})(?P<ext>.tif)"
        )

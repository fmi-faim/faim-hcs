from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from faim_hcs.stitching import Tile
from faim_hcs.stitching.Tile import TilePosition


class CellVoyagerStackedTile:
    def __init__(
        self,
        files: pd.DataFrame,
        shape: tuple[int, int],
        position: TilePosition,
        background_correction_matrices: dict[str, Union[Path, str]] = None,
        illumination_correction_matrices: dict[str, Union[Path, str]] = None,
    ):
        self.files = files.sort_values(by="ZIndex")
        self.shape = shape
        self.position = position
        self.background_correction_matrices = background_correction_matrices
        self.illumination_correction_matrices = illumination_correction_matrices

    def __repr__(self):
        return (
            f"Tile(path='{self.path}', shape={self.shape}, "
            f"position={self.position})"
        )

    def __str__(self):
        return self.__repr__()

    def get_yx_position(self) -> tuple[int, int]:
        return self.position.y, self.position.x

    def get_zyx_position(self) -> tuple[int, int, int]:
        return self.position.z, self.position.y, self.position.x

    def get_position(self) -> tuple[int, int, int, int, int]:
        return (
            self.position.time,
            self.position.channel,
            self.position.z,
            self.position.y,
            self.position.x,
        )

    def load_data(self) -> np.ndarray:
        tiles = [
            Tile(
                path=r["path"],
                shape=self.shape,
                position=TilePosition(
                    time=self.position.time,
                    channel=self.position.channel,
                    z=r["ZIndex"],
                    y=self.position.y,
                    x=self.position.x,
                ),
                background_correction_matrix_path=self.background_correction_matrices,
                illumination_correction_matrix_path=self.illumination_correction_matrices,
            )
            for i, r in self.files.iterrows()
        ]
        return np.stack([t.load_data() for t in tiles])

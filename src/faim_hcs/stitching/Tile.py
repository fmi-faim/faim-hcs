from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy._typing import NDArray
from pydantic import BaseModel, NonNegativeInt
from tifffile import imread


class TilePosition(BaseModel):
    time: Optional[NonNegativeInt]
    channel: Optional[NonNegativeInt]
    z: int
    y: int
    x: int

    def __repr__(self):
        return f"TilePosition(time={self.time}, channel={self.channel}, z={self.z}, y={self.y}, x={self.x})"

    def __str__(self):
        return self.__repr__()


class Tile:
    """
    A tile with a path to the image data, shape and position.
    """

    path: str
    shape: tuple[int, int]
    position: TilePosition
    background_correction_matrix_path: Optional[Union[Path, str]] = None
    illumination_correction_matrix_path: Optional[Union[Path, str]] = None

    def __init__(
        self,
        path: Union[Path, str],
        shape: tuple[int, int],
        position: TilePosition,
        background_correction_matrix_path: Optional[Union[Path, str]] = None,
        illumination_correction_matrix_path: Optional[Union[Path, str]] = None,
    ):
        super().__init__()
        self.path = path
        self.shape = shape
        self.position = position
        self.background_correction_matrix_path = background_correction_matrix_path
        self.illumination_correction_matrix_path = illumination_correction_matrix_path

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

    def load_data(self) -> NDArray:
        """
        Load the image data from the path.

        Returns
        -------
        Image data
        """
        data = imread(self.path)
        data = self._apply_background_correction(data)
        data = self._apply_illumination_correction(data)

        return data

    def _apply_illumination_correction(self, data):
        dtype = data.dtype
        if self.illumination_correction_matrix_path is not None:
            icm = imread(self.illumination_correction_matrix_path)
            assert icm.shape == data.shape, (
                f"Illumination correction matrix shape {icm.shape} "
                f"does not match image shape {data.shape}."
            )
            mi, ma = np.iinfo(dtype).min, np.iinfo(dtype).max
            data = np.clip(data / icm, a_min=mi, a_max=ma).astype(dtype)
        return data

    def _apply_background_correction(self, data):
        dtype = data.dtype
        if self.background_correction_matrix_path is not None:
            bgcm = imread(self.background_correction_matrix_path)
            assert bgcm.shape == data.shape, (
                f"Background correction matrix shape {bgcm.shape} "
                f"does not match image shape {data.shape}."
            )
            mi, ma = np.iinfo(dtype).min, np.iinfo(dtype).max
            data = np.clip(data - bgcm, a_min=mi, a_max=ma).astype(dtype)
        return data

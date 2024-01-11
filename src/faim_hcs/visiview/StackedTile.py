import tifffile
from numpy._typing import NDArray

from faim_hcs.stitching import Tile


class StackedTile(Tile):
    def load_data(self) -> NDArray:
        data = tifffile.memmap(self.path, mode="r")

        data = self._apply_background_correction(data)
        data = self._apply_illumination_correction(data)
        return data

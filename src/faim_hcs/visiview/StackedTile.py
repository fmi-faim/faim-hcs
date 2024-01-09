from numpy._typing import NDArray
from tifffile import TiffFile

from faim_hcs.stitching import Tile


class StackedTile(Tile):
    def load_data(self) -> NDArray:
        with TiffFile(self.path) as tif:
            data = tif.asarray()[self.position.z]

        data = self._apply_background_correction(data)
        data = self._apply_illumination_correction(data)
        return data

from abc import ABC
from copy import copy

import numpy as np

from faim_hcs.stitching import Tile, stitching_utils


class AbstractAlignment(ABC):
    _unaligned_tiles: list[Tile] = None
    _aligned_tiles: list[Tile] = None

    def __init__(self, tiles: list[Tile]) -> None:
        super().__init__()
        self._unaligned_tiles = stitching_utils.shift_to_origin(tiles)
        self._aligned_tiles = self._align(self._unaligned_tiles)

    def _align(self, tiles: list[Tile]) -> list[Tile]:
        raise NotImplementedError

    def get_tiles(self) -> list[Tile]:
        return self._aligned_tiles


class StageAlignment(AbstractAlignment):
    """
    Align tiles using stage positions.
    """

    def _align(self, tiles: list[Tile]) -> list[Tile]:
        return tiles


class GridAlignment(AbstractAlignment):
    """
    Align tiles on a regular grid.
    """

    def _align(self, tiles: list[Tile]) -> list[Tile]:
        aligned_tiles = []

        tile_shape = tiles[0].shape

        grid_positions_y = set()
        grid_positions_x = set()
        tile_map = {}
        for tile in tiles:
            assert (
                tile.shape[-2:] == tile_shape[-2:]
            ), "All tiles must have the same YX shape."
            y_pos = int(np.round(tile.position.y / tile_shape[-2]))
            x_pos = int(np.round(tile.position.x / tile_shape[-1]))
            if (y_pos, x_pos) in tile_map.keys():
                tile_map[(y_pos, x_pos)].append(tile)
            else:
                tile_map[(y_pos, x_pos)] = [tile]
            grid_positions_y.add(y_pos)
            grid_positions_x.add(x_pos)

        grid_positions_y = sorted(grid_positions_y)
        grid_positions_x = sorted(grid_positions_x)
        for y_pos in grid_positions_y:
            for x_pos in grid_positions_x:
                if (y_pos, x_pos) in tile_map.keys():
                    for unaligned_tile in tile_map[(y_pos, x_pos)]:
                        new_tile = copy(unaligned_tile)
                        new_tile.position.y = y_pos * tile_shape[-2]
                        new_tile.position.x = x_pos * tile_shape[-1]
                        aligned_tiles.append(new_tile)

        return aligned_tiles

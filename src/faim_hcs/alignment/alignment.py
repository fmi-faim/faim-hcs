from abc import ABC

from faim_hcs.stitching import Tile, stitching_utils
from faim_hcs.stitching.Tile import TilePosition


class AbstractAlignment(ABC):
    _unaligned_tiles: list[Tile] = None
    _aligned_tiles: list[Tile] = None

    def __init__(self, tiles: list[Tile]) -> None:
        super().__init__()
        self._unaligned_tiles = stitching_utils.shift_to_origin(tiles)
        self._aligned_tiles = self._align(tiles)

    def _align(self, tiles: list[Tile]) -> list[Tile]:
        raise NotImplementedError()

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
            assert tile.shape == tile_shape, "All tiles must have the same shape."
            y_pos = tile.position.y // tile_shape[0]
            x_pos = tile.position.x // tile_shape[1]
            if (y_pos, x_pos) in tile_map.keys():
                tile_map[(y_pos, x_pos)].append(tile)
            else:
                tile_map[(y_pos, x_pos)] = [tile]
            grid_positions_y.add(y_pos)
            grid_positions_x.add(x_pos)

        grid_positions_y = list(sorted(grid_positions_y))
        grid_positions_x = list(sorted(grid_positions_x))
        for y_pos in grid_positions_y:
            for x_pos in grid_positions_x:
                if (y_pos, x_pos) in tile_map.keys():
                    for unaligned_tile in tile_map[(y_pos, x_pos)]:
                        aligned_tiles.append(
                            Tile(
                                path=unaligned_tile.path,
                                shape=unaligned_tile.shape,
                                position=TilePosition(
                                    time=unaligned_tile.position.time,
                                    channel=unaligned_tile.position.channel,
                                    z=unaligned_tile.position.z,
                                    y=y_pos * tile_shape[0],
                                    x=x_pos * tile_shape[1],
                                ),
                                background_correction_matrix_path=unaligned_tile.background_correction_matrix_path,
                                illumination_correction_matrix_path=unaligned_tile.illumination_correction_matrix_path,
                            )
                        )

        return aligned_tiles

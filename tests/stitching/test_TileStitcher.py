import numpy as np
import pytest
from numpy.testing import assert_array_equal

from faim_hcs.stitching import DaskTileStitcher, Tile, stitching_utils
from faim_hcs.stitching.Tile import TilePosition


@pytest.fixture
def tiles():
    tiles = [
        Tile(
            path="path1",
            shape=(10, 10),
            position=TilePosition(time=0, channel=0, z=0, y=0, x=0),
        ),
        Tile(
            path="path2",
            shape=(10, 10),
            position=TilePosition(time=0, channel=0, z=0, y=0, x=10),
        ),
        Tile(
            path="path3",
            shape=(10, 10),
            position=TilePosition(time=0, channel=0, z=0, y=10, x=0),
        ),
        Tile(
            path="path4",
            shape=(10, 10),
            position=TilePosition(time=0, channel=0, z=0, y=10, x=10),
        ),
    ]

    for i, tile in enumerate(tiles):
        tile.i = i

        def loader(self=tile):
            return np.ones(self.shape) * self.i

        tile.load_data = loader
    return tiles


def test_block_to_tile_map(tiles):
    """
    Test mapping of chunk-blocks to tiles if the chunk shape matches exactly
    the tile shape.
    """
    ts = DaskTileStitcher(
        tiles=tiles,
        chunk_shape=(10, 10),
    )

    assert ts._shape == (1, 1, 1, 20, 20)
    assert len(ts._block_to_tile_map) == 4
    assert ts._block_to_tile_map[(0, 0, 0, 0, 0)] == [tiles[0]]
    assert ts._block_to_tile_map[(0, 0, 0, 0, 1)] == [tiles[1]]
    assert ts._block_to_tile_map[(0, 0, 0, 1, 0)] == [tiles[2]]
    assert ts._block_to_tile_map[(0, 0, 0, 1, 1)] == [tiles[3]]


def test_block_to_tile_map_large_chunks(tiles):
    """
    Test mapping of chunk-blocks to tiles if the chunk shape is larger than
    the tile shape.
    """
    ts = DaskTileStitcher(
        tiles=tiles,
        chunk_shape=(15, 15),
    )

    assert ts._shape == (1, 1, 1, 20, 20)
    assert len(ts._block_to_tile_map) == 4
    assert ts._block_to_tile_map[(0, 0, 0, 0, 0)] == [
        tiles[0],
        tiles[1],
        tiles[2],
        tiles[3],
    ]
    assert ts._block_to_tile_map[(0, 0, 0, 0, 1)] == [tiles[1], tiles[3]]
    assert ts._block_to_tile_map[(0, 0, 0, 1, 0)] == [tiles[2], tiles[3]]
    assert ts._block_to_tile_map[(0, 0, 0, 1, 1)] == [tiles[3]]


def test_block_to_tile_map_small_chunks(tiles):
    """
    Test mapping of chunk-blocks to tiles if the chunk shape is smaller than
    the tile shape.
    """
    ts = DaskTileStitcher(
        tiles=tiles,
        chunk_shape=(8, 8),
    )

    assert ts._shape == (1, 1, 1, 20, 20)
    assert len(ts._block_to_tile_map) == 9
    assert ts._block_to_tile_map[(0, 0, 0, 0, 0)] == [tiles[0]]
    assert ts._block_to_tile_map[(0, 0, 0, 0, 1)] == [tiles[0], tiles[1]]
    assert ts._block_to_tile_map[(0, 0, 0, 0, 2)] == [tiles[1]]
    assert ts._block_to_tile_map[(0, 0, 0, 1, 0)] == [tiles[0], tiles[2]]
    assert ts._block_to_tile_map[(0, 0, 0, 1, 1)] == [
        tiles[0],
        tiles[1],
        tiles[2],
        tiles[3],
    ]
    assert ts._block_to_tile_map[(0, 0, 0, 1, 2)] == [tiles[1], tiles[3]]
    assert ts._block_to_tile_map[(0, 0, 0, 2, 0)] == [tiles[2]]
    assert ts._block_to_tile_map[(0, 0, 0, 2, 1)] == [tiles[2], tiles[3]]
    assert ts._block_to_tile_map[(0, 0, 0, 2, 2)] == [tiles[3]]


def test_stitch_exact(tiles):
    ts = DaskTileStitcher(tiles=tiles, chunk_shape=(7, 7))
    stitched = ts.get_stitched_image(
        transform_func=stitching_utils.translate_tiles_2d,
        fuse_func=stitching_utils.fuse_sum,
    )
    assert stitched.shape == (1, 1, 1, 20, 20)

    assert_array_equal(stitched[..., :10, :10], np.ones((1, 1, 1, 10, 10)) * 0)
    assert_array_equal(stitched[..., :10, 10:20], np.ones((1, 1, 1, 10, 10)) * 1)
    assert_array_equal(stitched[..., 10:20, :10], np.ones((1, 1, 1, 10, 10)) * 2)
    assert_array_equal(stitched[..., 10:20, 10:20], np.ones((1, 1, 1, 10, 10)) * 3)


@pytest.fixture
def overlapping_tiles():
    tiles = [
        Tile(
            path="path1",
            shape=(10, 10),
            position=TilePosition(time=0, channel=0, z=0, y=0, x=0),
        ),
        Tile(
            path="path2",
            shape=(10, 10),
            position=TilePosition(time=0, channel=0, z=0, y=0, x=5),
        ),
        Tile(
            path="path3",
            shape=(10, 10),
            position=TilePosition(time=0, channel=0, z=0, y=5, x=0),
        ),
        Tile(
            path="path4",
            shape=(10, 10),
            position=TilePosition(time=0, channel=0, z=0, y=5, x=5),
        ),
    ]
    for i, tile in enumerate(tiles):
        tile.i = i

        def loader(self=tile):
            return np.ones(self.shape) * self.i

        tile.load_data = loader

    return tiles


def test_stitch_overlapping(overlapping_tiles):
    ts = DaskTileStitcher(tiles=overlapping_tiles, chunk_shape=(5, 7))
    stitched = ts.get_stitched_image(
        transform_func=stitching_utils.translate_tiles_2d,
        fuse_func=stitching_utils.fuse_sum,
    )
    assert stitched.shape == (1, 1, 1, 15, 15)
    assert_array_equal(stitched[..., :5, :5], np.ones((1, 1, 1, 5, 5)) * 0)
    assert_array_equal(stitched[..., :5, 5:15], np.ones((1, 1, 1, 5, 10)) * 1)
    assert_array_equal(stitched[..., 5:15, :5], np.ones((1, 1, 1, 10, 5)) * 2)
    assert_array_equal(stitched[..., 5:10, 5:10], np.ones((1, 1, 1, 5, 5)) * 6)
    assert_array_equal(stitched[..., 10:15, 5:10], np.ones((1, 1, 1, 5, 5)) * 5)
    assert_array_equal(stitched[..., 5:10, 10:15], np.ones((1, 1, 1, 5, 5)) * 4)
    assert_array_equal(stitched[..., 10:15, 10:15], np.ones((1, 1, 1, 5, 5)) * 3)

from copy import copy

import numpy as np
from numpy._typing import ArrayLike
from skimage.transform import EuclideanTransform, warp

from faim_hcs.stitching.Tile import Tile, TilePosition


def fuse_mean(warped_tiles: ArrayLike, warped_masks: ArrayLike) -> ArrayLike:
    """
    Fuse transformed tiles and compute the mean of the overlapping pixels.

    Parameters
    ----------
    warped_tiles :
        Tile images transformed to the final image space.
    warped_masks :
        Masks indicating foreground pixels for the transformed tiles.

    Returns
    -------
    Fused image.
    """
    weights = warped_masks.astype(np.float32)
    weights = weights / weights.sum(axis=0)

    fused_image = np.sum(warped_tiles * weights, axis=0)
    return fused_image.astype(warped_tiles.dtype)


def fuse_sum(warped_tiles: ArrayLike, warped_masks: ArrayLike) -> ArrayLike:
    """
    Fuse transformed tiles and compute the sum of the overlapping pixels.

    Parameters
    ----------
    warped_tiles :
        Tile images transformed to the final image space.
    warped_masks :
        Masks indicating foreground pixels for the transformed tiles.

    Returns
    -------
    Fused image.
    """
    fused_image = np.sum(warped_tiles, axis=0)
    return fused_image.astype(warped_tiles.dtype)


def translate_tiles_2d(block_info, yx_chunk_shape, dtype, tiles):
    array_location = block_info[None]["array-location"]
    chunk_yx_origin = np.array([array_location[3][0], array_location[4][0]])
    warped_tiles = np.zeros((len(tiles),) + yx_chunk_shape, dtype=dtype)
    warped_masks = np.zeros_like(warped_tiles, dtype=bool)
    for i, tile in enumerate(tiles):
        tile_origin = np.array(tile.get_yx_position())
        transform = EuclideanTransform(
            translation=(chunk_yx_origin - tile_origin)[::-1]
        )
        tile_data = tile.load_data()
        mask = np.ones(tile_data.shape, dtype=bool)
        warped_tiles[
            i,
            ...,
        ] = warp(
            tile_data,
            transform,
            cval=0,
            output_shape=yx_chunk_shape,
            order=0,
            preserve_range=True,
        ).astype(dtype)

        warped_masks[i] = warp(
            mask,
            transform,
            cval=False,
            output_shape=yx_chunk_shape,
            order=0,
            preserve_range=True,
        ).astype(bool)
    return warped_tiles, warped_masks


def assemble_chunk(
    block_info=None, tile_map=None, warp_func=None, fuse_func=None, dtype=None
):
    chunk_location = block_info[None]["chunk-location"]
    chunk_shape = block_info[None]["chunk-shape"]
    tiles = tile_map[chunk_location]

    if len(tiles) > 0:
        warped_tiles, warped_masks = warp_func(
            block_info, chunk_shape[-2:], dtype, tiles
        )

        stitched_img = fuse_func(
            warped_tiles,
            warped_masks,
        ).astype(dtype=dtype)
        stitched_img = stitched_img[np.newaxis, np.newaxis, np.newaxis, ...]
    else:
        stitched_img = np.zeros(chunk_shape, dtype=dtype)

    return stitched_img


def shift_to_origin(tiles: list[Tile]) -> list[Tile]:
    """
    Shift tile positions such that the minimal position is (0, 0, 0, 0, 0).

    Parameters
    ----------
    tiles :
        List of tiles.

    Returns
    -------
    List of shifted tiles.
    """
    min_tile_origin = np.min([np.array(tile.get_position()) for tile in tiles], axis=0)
    shifted_tiles = copy(tiles)
    for tile in shifted_tiles:
        shifted_pos = np.array(tile.get_position()) - min_tile_origin
        tile.position = TilePosition(
            time=shifted_pos[0],
            channel=shifted_pos[1],
            z=shifted_pos[2],
            y=shifted_pos[3],
            x=shifted_pos[4],
        )
    return shifted_tiles

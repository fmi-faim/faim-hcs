from copy import copy

import numpy as np
from numpy._typing import NDArray
from scipy.ndimage import distance_transform_edt

from faim_hcs.stitching.Tile import Tile, TilePosition


def fuse_linear(warped_tiles: NDArray, warped_masks: NDArray) -> NDArray:
    """
    Fuse transformed tiles using a linear gradient to compute the weighted
    average where tiles are overlapping.

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
    dtype = warped_tiles.dtype
    if warped_tiles.shape[0] > 1:
        warped_masks = warped_masks[:, 0]
        weights = np.zeros_like(warped_masks, dtype=np.float32)
        for i, mask in enumerate(warped_masks):
            weights[i] = distance_transform_edt(
                warped_masks[i].astype(np.float32),
            )

        denominator = weights.sum(axis=0)
        weights = np.true_divide(weights, denominator, where=denominator > 0)
        weights = np.nan_to_num(weights, nan=0, posinf=1, neginf=0)
        weights = np.clip(
            weights,
            0,
            1,
        )
        weights = weights[:, np.newaxis, ...]
    else:
        weights = warped_masks

    return np.sum(warped_tiles * weights, axis=0).astype(dtype)


def fuse_mean(warped_tiles: NDArray, warped_masks: NDArray) -> NDArray:
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
    warped_masks = warped_masks[:, 0]
    denominator = warped_masks.sum(axis=0)
    weights = np.true_divide(warped_masks, denominator, where=denominator > 0)
    weights = np.clip(
        np.nan_to_num(weights, nan=0, posinf=1, neginf=0),
        0,
        1,
    )
    weights = weights[:, np.newaxis, ...]

    fused_image = np.sum(warped_tiles * weights, axis=0)
    return fused_image.astype(warped_tiles.dtype)


def fuse_sum(warped_tiles: NDArray, warped_masks: NDArray) -> NDArray:
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


def translate_tiles_2d(block_info, chunk_shape, tiles):
    """
    Translate tiles to their relative position inside the given block.

    Parameters
    ----------
    block_info :
        da.map_blocks block_info.
    chunk_shape :
        shape of the chunk in zyx.
    tiles :
        list of tiles.

    Returns
    -------
        translated tiles, translated masks
    """
    array_location = block_info[None]["array-location"]
    chunk_zyx_origin = np.array(
        [array_location[2][0], array_location[3][0], array_location[4][0]]
    )
    warped_tiles = []
    warped_masks = []
    for tile in tiles:
        tile_origin = np.array(tile.get_zyx_position())
        tile_data = tile.load_data()
        if tile_data.ndim == 2:
            tile_data = tile_data[np.newaxis, ...]
        warped_mask, warped_tile = shift_yx(
            chunk_zyx_origin, tile_data, tile_origin, chunk_shape
        )

        warped_tiles.append(warped_tile)
        warped_masks.append(warped_mask)

    return np.array(warped_tiles), np.array(warped_masks)


def shift_yx(chunk_zyx_origin, tile_data, tile_origin, chunk_shape):
    warped_tile = np.zeros(chunk_shape, dtype=tile_data.dtype)
    warped_mask = np.zeros(chunk_shape, dtype=bool)
    yx_shift = (tile_origin - chunk_zyx_origin)[1:]
    if yx_shift[0] < 0:
        tile_start_y = abs(yx_shift[0])
        tile_end_y = min(tile_start_y + chunk_shape[1], tile_data.shape[1])
    else:
        tile_start_y = 0
        tile_end_y = max(
            0, min(tile_start_y + chunk_shape[1] - yx_shift[0], tile_data.shape[1])
        )
    if yx_shift[1] < 0:
        tile_start_x = abs(yx_shift[1])
        tile_end_x = min(tile_start_x + chunk_shape[2], tile_data.shape[2])
    else:
        tile_start_x = 0
        tile_end_x = min(
            tile_start_x + chunk_shape[2] - yx_shift[1], tile_data.shape[2]
        )
    tile_data = tile_data[:, tile_start_y:tile_end_y, tile_start_x:tile_end_x]
    if tile_data.size > 0:
        start_y = max(0, yx_shift[0])
        end_y = start_y + tile_data.shape[1]
        start_x = max(0, yx_shift[1])
        end_x = start_x + tile_data.shape[2]
        warped_tile[..., start_y:end_y, start_x:end_x] = tile_data
        warped_mask[..., start_y:end_y, start_x:end_x] = True
    return warped_mask, warped_tile


def assemble_chunk(
    block_info=None, tile_map=None, warp_func=None, fuse_func=None, dtype=None
):
    """
    Assemble a chunk of the stitched image.

    Parameters
    ----------
    block_info :
        da.map_blocks block_info.
    tile_map :
        map of block positions to tiles.
    warp_func :
        function used to warp tiles.
    fuse_func :
        function used to fuse tiles.
    dtype :
        tile data type.

    Returns
    -------
        fused tiles corresponding to this block/chunk
    """
    chunk_location = block_info[None]["chunk-location"]
    chunk_shape = block_info[None]["chunk-shape"]
    tiles = tile_map[chunk_location]

    if len(tiles) > 0:
        warped_tiles, warped_masks = warp_func(block_info, chunk_shape[-3:], tiles)

        if len(tiles) > 1:
            stitched_img = fuse_func(
                warped_tiles,
                warped_masks,
            )
            stitched_img = stitched_img[np.newaxis, np.newaxis, ...]
        else:
            stitched_img = warped_tiles[np.newaxis, ...]
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

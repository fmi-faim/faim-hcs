import os
from os.path import join
from pathlib import Path
from typing import Callable, Optional, Union

import dask.array as da
import zarr
from dask.distributed import Client, wait
from numcodecs import Blosc
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.writer import (
    _get_valid_axes,
    write_multiscales_metadata,
    write_plate_metadata,
    write_well_metadata,
)
from pydantic import BaseModel

from faim_hcs import dask_utils
from faim_hcs.dask_utils import LocalClusterFactory
from faim_hcs.hcs.acquisition import PlateAcquisition
from faim_hcs.hcs.plate import PlateLayout, get_rows_and_columns
from faim_hcs.stitching import stitching_utils


class NGFFPlate(BaseModel):
    root_dir: Union[Path, str]
    name: str
    layout: PlateLayout
    order_name: str
    barcode: str


class ConvertToNGFFPlate:
    """
    Convert a plate acquisition to an NGFF plate.
    """

    _ngff_plate: NGFFPlate

    def __init__(
        self,
        ngff_plate: NGFFPlate,
        yx_binning: int = 1,
        stitching_yx_chunk_size_factor: int = 1,
        warp_func: Callable = stitching_utils.translate_tiles_2d,
        fuse_func: Callable = stitching_utils.fuse_mean,
        client: Client = None,
    ):
        """
        Parameters
        ----------
        ngff_plate :
            NGFF plate information.
        yx_binning :
            YX binning factor.
        warp_func :
            Function used to warp tile images.
        fuse_func :
            Function used to fuse tile images.
        client :
            Dask client used for the conversion.
        """
        assert (
            isinstance(yx_binning, int) and yx_binning >= 1
        ), "yx_binning must be an integer >= 1."
        self._ngff_plate = ngff_plate
        self._yx_binning = yx_binning
        self._stitching_yx_chunk_size_factor = stitching_yx_chunk_size_factor
        self._warp_func = warp_func
        self._fuse_func = fuse_func
        if client is None:
            self._cluster_factory = LocalClusterFactory()
            self._client = self._cluster_factory.get_client()
        else:
            self._cluster_factory = None
            self._client = client

    def _create_zarr_plate(
        self, plate_acquisition: PlateAcquisition, wells: Optional[list[str]] = None
    ) -> zarr.Group:
        plate_path = join(self._ngff_plate.root_dir, self._ngff_plate.name + ".zarr")
        if not os.path.exists(plate_path):
            os.makedirs(plate_path, exist_ok=False)
            store = parse_url(plate_path, mode="w").store
            plate = zarr.group(store=store)

            rows, cols = get_rows_and_columns(layout=self._ngff_plate.layout)

            write_plate_metadata(
                plate,
                columns=cols,
                rows=rows,
                wells=[
                    f"{w[0]}/{w[1:]}" for w in plate_acquisition.get_well_names(wells)
                ],
                name=self._ngff_plate.name,
                field_count=1,
            )

            attrs = plate.attrs.asdict()
            attrs["order_name"] = self._ngff_plate.order_name
            attrs["barcode"] = self._ngff_plate.barcode
            plate.attrs.put(attrs)
            return plate
        else:
            store = parse_url(plate_path, mode="w").store
            return zarr.group(store=store)

    def run(
        self,
        plate_acquisition: PlateAcquisition,
        wells: list[str] = None,
        well_sub_group: str = "0",
        chunks: Union[tuple[int, int], tuple[int, int, int]] = (2048, 2048),
        max_layer: int = 3,
        storage_options: dict = None,
    ):
        """
        Convert a plate acquisition to an NGFF plate.

        Parameters
        ----------
        plate_acquisition :
            A single plate acquisition.
        well_sub_group :
            Name of the well subgroup.
        chunks :
            Chunk size in (Z)YX.
        max_layer :
            Maximum layer of the resolution pyramid layers.
        storage_options :
            Zarr storage options.

        Returns
        -------
            zarr.Group of the plate.
        """
        assert 2 <= len(chunks) <= 3, "Chunks must be 2D or 3D."
        plate = self._create_zarr_plate(plate_acquisition, wells=wells)
        well_acquisitions = plate_acquisition.get_well_acquisitions(wells)

        for well_acquisition in well_acquisitions:
            well_group = self._create_well_group(
                plate,
                well_acquisition,
                well_sub_group,
            )
            group = well_group[well_sub_group]
            self._write_stitched_image(
                group,
                chunks,
                plate_acquisition,
                storage_options,
                well_acquisition,
            )
            shapes, datasets = self._build_pyramid(
                group,
                chunks,
                max_layer,
                storage_options,
            )
            self._write_metadata(
                group, max_layer, shapes, datasets, plate_acquisition, well_acquisition
            )

        return plate

    def _write_metadata(
        self, group, max_layer, shapes, datasets, plate_acquisition, well_acquisition
    ):
        coordinate_transformations = well_acquisition.get_coordinate_transformations(
            max_layer=max_layer,
            yx_binning=self._yx_binning,
        )
        fmt = CurrentFormat()
        dims = len(shapes[0])
        fmt.validate_coordinate_transformations(
            dims, len(datasets), coordinate_transformations
        )
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform
        axes = _get_valid_axes(dims, well_acquisition.get_axes(), fmt)
        write_multiscales_metadata(
            group,
            datasets,
            fmt,
            axes,
        )
        group.attrs["omero"] = {
            "channels": plate_acquisition.get_omero_channel_metadata()
        }
        group.attrs["acquisition_metadata"] = {
            "channels": [
                ch_metadata.dict()
                for ch_metadata in plate_acquisition.get_channel_metadata().values()
            ]
        }

    def _write_stitched_image(
        self,
        group,
        chunks,
        plate_acquisition,
        storage_options,
        well_acquisition,
    ):
        stitched_well_da = self._stitch_well_image(
            chunks,
            well_acquisition,
            output_shape=plate_acquisition.get_common_well_shape(),
        )
        binned_da = self._bin_yx(stitched_well_da).squeeze()
        rechunked_da = binned_da.rechunk(self._out_chunks(binned_da.shape, chunks))
        options = self._get_storage_options(storage_options, rechunked_da.shape, chunks)
        wait(
            self._client.persist(
                da.to_zarr(
                    arr=rechunked_da,
                    url=group.store,
                    compute=False,
                    component=str(Path(group.path, "0")),
                    storage_options=options,
                    compressor=options.get(
                        "compressor", zarr.storage.default_compressor
                    ),
                    dimension_separator=group._store._dimension_separator,
                ),
            )
        )

    def _build_pyramid(
        self,
        group,
        chunks,
        max_layer,
        storage_options,
    ):
        image = da.from_zarr(url=group.store, component=str(Path(group.path, "0")))
        datasets = [{"path": "0"}]
        shapes = [image.shape]
        for path in range(1, max_layer + 1):
            image = da.coarsen(
                reduction=dask_utils.mean_cast_to(image.dtype),
                x=image,
                axes={
                    image.ndim - 2: 2,
                    image.ndim - 1: 2,
                },
                trim_excess=True,
            )
            options = self._get_storage_options(storage_options, image.shape, chunks)
            image = image.rechunk(options["chunks"])
            wait(
                self._client.persist(
                    da.to_zarr(
                        arr=image,
                        url=group.store,
                        compute=False,
                        component=str(Path(group.path, str(path))),
                        storage_options=options,
                        compressor=options.get(
                            "compressor", zarr.storage.default_compressor
                        ),
                        dimension_separator=group._store._dimension_separator,
                    )
                )
            )
            datasets.append({"path": str(path)})
            shapes.append(image.shape)
            image = da.from_zarr(
                url=group.store, component=str(Path(group.path, str(path)))
            )

        return shapes, datasets

    def _bin_yx(self, image_da):
        if self._yx_binning > 1:
            return da.coarsen(
                reduction=dask_utils.mean_cast_to(image_da.dtype),
                x=image_da,
                axes={
                    0: 1,
                    1: 1,
                    2: 1,
                    3: self._yx_binning,
                    4: self._yx_binning,
                },
                trim_excess=True,
            )
        else:
            return image_da

    def _stitch_well_image(
        self,
        chunks,
        well_acquisition,
        output_shape: tuple[int, int, int, int, int],
    ):
        from faim_hcs.stitching import DaskTileStitcher

        stitcher = DaskTileStitcher(
            tiles=well_acquisition.get_tiles(),
            yx_chunk_shape=(
                chunks[-2],
                chunks[-1],
            ),
            output_shape=output_shape,
            dtype=well_acquisition.get_dtype(),
        )
        image_da = stitcher.get_stitched_dask_array(
            warp_func=self._warp_func,
            fuse_func=self._fuse_func,
        )
        return image_da

    def _create_well_group(self, plate, well_acquisition, well_sub_group):
        row, col = well_acquisition.get_row_col()
        well_group = plate.require_group(row).require_group(col)
        well_group.require_group(well_sub_group)
        write_well_metadata(well_group, [{"path": well_sub_group}])
        return well_group

    @staticmethod
    def _get_storage_options(
        storage_options: dict,
        output_shape: tuple[int, ...],
        chunks: tuple[int, ...],
    ):
        if storage_options is None:
            return dict(
                dimension_separator="/",
                compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
                chunks=ConvertToNGFFPlate._out_chunks(output_shape, chunks),
                write_empty_chunks=False,
            )
        else:
            return storage_options

    @staticmethod
    def _out_chunks(shape, chunks):
        if len(shape) == len(chunks):
            return chunks
        else:
            return (1,) * (len(shape) - len(chunks)) + chunks

    def __del__(self):
        if self._cluster_factory is not None:
            self._cluster_factory.__del__()

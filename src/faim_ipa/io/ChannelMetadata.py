from typing import Optional

from pydantic import BaseModel, NonNegativeInt, PositiveFloat


class ChannelMetadata(BaseModel):
    channel_index: NonNegativeInt
    channel_name: str
    display_color: str
    spatial_calibration_x: PositiveFloat
    spatial_calibration_y: PositiveFloat
    spatial_calibration_units: str
    z_spacing: Optional[PositiveFloat] = None
    wavelength: Optional[NonNegativeInt] = None
    exposure_time: Optional[PositiveFloat] = None
    exposure_time_unit: Optional[str] = None
    objective: str

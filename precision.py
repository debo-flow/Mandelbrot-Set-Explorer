"""
Numerical Precision Diagnostics Module.
Evaluates floating-point limits and grid sampling resolution for Deep Zoom operations.
"""
import math
import numpy as np
from dataclasses import dataclass
from enum import Enum

class PrecisionState(Enum):
    NORMAL = "Normal"
    WARNING = "Warning"
    LIMITATION = "Limitation Reached"

@dataclass
class PrecisionDiagnostics:
    state: PrecisionState
    horizontal_step: float
    vertical_step: float
    re_spacing: float
    im_spacing: float
    message: str

def evaluate_precision(
    re_min: float, re_max: float, im_min: float, im_max: float, width: int, height: int
) -> PrecisionDiagnostics:
    """
    Analyzes the requested coordinate grid against float64 IEEE-754 limitations.
    Calculates whether the complex-plane spacing between adjacent pixels
    exceeds the resolving capability of standard 64-bit floating point numbers.
    """
    if width <= 1 or height <= 1:
        raise ValueError("Dimensions must be greater than 1 to calculate step sizes.")
        
    for v in (re_min, re_max, im_min, im_max):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Coordinates must be finite numbers.")

    # Grid spacing (distance in the complex plane between adjacent pixels)
    re_step = (re_max - re_min) / (width - 1)
    im_step = (im_max - im_min) / (height - 1)

    # Local floating point resolution at the boundaries
    # The largest float magnitude in the range dictates the worst-case spacing
    max_re_mag = max(abs(re_min), abs(re_max))
    max_im_mag = max(abs(im_min), abs(im_max))

    # np.spacing returns the distance to the nearest representable float64
    re_spacing = np.spacing(max_re_mag)
    im_spacing = np.spacing(max_im_mag)

    state = PrecisionState.NORMAL
    msg = "Sampling precision is optimal for float64."

    # If the grid step is smaller than or equal to the float resolution, 
    # adjacent pixels will map to the exact same numerical coordinate.
    if re_step <= re_spacing or im_step <= im_spacing:
        state = PrecisionState.LIMITATION
        msg = "Precision limitation detected: multiple adjacent pixel coordinates map to identical floating-point values."
    # If we are within two orders of magnitude of the limit, precision degradation (banding) is visible.
    elif re_step <= 100 * re_spacing or im_step <= 100 * im_spacing:
        state = PrecisionState.WARNING
        msg = "Warning: neighboring horizontal or vertical sample coordinates are approaching float64 limits. Image may exhibit blockiness."

    return PrecisionDiagnostics(
        state=state,
        horizontal_step=re_step,
        vertical_step=im_step,
        re_spacing=re_spacing,
        im_spacing=im_spacing,
        message=msg
    )

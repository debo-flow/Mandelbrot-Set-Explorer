"""
Mandelbrot Computation Engine.
Handles the mathematical generation of the Mandelbrot Set using escape-time algorithms
and smooth continuous coloring approximation.
"""
import math
import numpy as np
from typing import Tuple

def compute_mandelbrot(
    re_min: float = -2.5,
    re_max: float = 1.0,
    im_min: float = -1.5,
    im_max: float = 1.5,
    width: int = 800,
    height: int = 800,
    max_iter: int = 100,
    escape_radius: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Mandelbrot Set using an optimized escape-time algorithm.

    Args:
        re_min, re_max: Boundaries of the real axis.
        im_min, im_max: Boundaries of the imaginary axis.
        width: Pixel width of the output image.
        height: Pixel height of the output image.
        max_iter: Maximum iterations before classifying as unresolved.
        escape_radius: Radius threshold for escape (must be >= 2.0 for standard algorithms).

    Returns:
        escape_times (np.ndarray): 2D integer array recording the iteration at which a point escaped.
        smooth_escape_times (np.ndarray): 2D float array containing continuous escape approximations.
        escaped_mask (np.ndarray): 2D boolean array where True means the point escaped.
        unresolved_mask (np.ndarray): 2D boolean array where True means the point did not escape.
        c_matrix (np.ndarray): 2D complex coordinate grid.
    """
    # 1. Parameter Validation
    for v in (re_min, re_max, im_min, im_max):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Coordinate boundaries must be finite numbers.")
            
    if re_max <= re_min:
        raise ValueError("re_max must be strictly greater than re_min")
    if im_max <= im_min:
        raise ValueError("im_max must be strictly greater than im_min")
    if width <= 0 or height <= 0:
        raise ValueError("Grid width and height must be positive integers")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if escape_radius < 2.0:
        raise ValueError("escape_radius must be >= 2.0 for mathematical stability.")

    # 2. Coordinate Grid Generation
    x = np.linspace(re_min, re_max, width)
    y = np.linspace(im_min, im_max, height)
    X, Y = np.meshgrid(x, y)
    c_matrix = X + 1j * Y

    # 3. Initialization
    z = np.zeros_like(c_matrix, dtype=np.complex128)
    escape_times = np.zeros(c_matrix.shape, dtype=np.int32)
    smooth_escape_times = np.zeros(c_matrix.shape, dtype=np.float64)
    
    active = np.ones(c_matrix.shape, dtype=bool)

    # 4. Escape-time Iteration
    for i in range(max_iter):
        if not np.any(active):
            break

        z[active] = z[active]**2 + c_matrix[active]

        # Detect active points that exceeded the escape radius
        escaped_now = np.abs(z[active]) > escape_radius
        
        if np.any(escaped_now):
            escaped_full = np.zeros_like(active)
            escaped_full[active] = escaped_now

            # Record standard integer iteration
            escape_times[escaped_full] = i + 1

            # Record continuous/smooth coloring approximation:
            # mu = n + 1 - log2(ln(|z|))
            # Guaranteed safe since |z| > escape_radius >= 2.0, so ln(|z|) > ln(2) > 0.
            z_esc = np.abs(z[escaped_full])
            log_z = np.log(z_esc)
            smooth_escape_times[escaped_full] = (i + 1) - np.log2(log_z)

            active[escaped_full] = False

    # 5. Classify final states
    unresolved_mask = active.copy()
    escaped_mask = ~active

    return escape_times, smooth_escape_times, escaped_mask, unresolved_mask, c_matrix

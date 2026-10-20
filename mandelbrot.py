"""
Mandelbrot Computation Engine.
Handles the mathematical generation of the Mandelbrot Set using an escape-time algorithm.
"""
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Mandelbrot Set using an optimized escape-time algorithm.

    Args:
        re_min, re_max: Boundaries of the real axis.
        im_min, im_max: Boundaries of the imaginary axis.
        width: Pixel width of the output image.
        height: Pixel height of the output image.
        max_iter: Maximum number of iterations before assuming a point is unresolved.
        escape_radius: Radius beyond which an orbit is guaranteed to escape.

    Returns:
        escape_times (np.ndarray): 2D integer array recording the iteration at which a point escaped.
        escaped_mask (np.ndarray): 2D boolean array where True means the point escaped.
        unresolved_mask (np.ndarray): 2D boolean array where True means the point did not escape.
        c_matrix (np.ndarray): 2D complex array of the coordinate grid.
    """
    # 1. Parameter Validation
    if re_max <= re_min:
        raise ValueError("re_max must be strictly greater than re_min")
    if im_max <= im_min:
        raise ValueError("im_max must be strictly greater than im_min")
    if width <= 0 or height <= 0:
        raise ValueError("Grid width and height must be positive integers")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if escape_radius <= 0:
        raise ValueError("escape_radius must be positive")

    # 2. Coordinate Grid Generation
    x = np.linspace(re_min, re_max, width)
    y = np.linspace(im_min, im_max, height)
    X, Y = np.meshgrid(x, y)
    
    # Complex coordinate matrix c = x + iy
    c_matrix = X + 1j * Y

    # 3. Initialization
    z = np.zeros_like(c_matrix, dtype=np.complex128)
    escape_times = np.zeros(c_matrix.shape, dtype=np.int32)
    
    # Active mask tracks points that have not yet escaped
    active = np.ones(c_matrix.shape, dtype=bool)

    # 4. Escape-time Iteration
    for i in range(max_iter):
        if not np.any(active):
            break  # Stop early if all points have escaped

        # z_(n+1) = z_n^2 + c (Computed only for active points)
        z[active] = z[active]**2 + c_matrix[active]

        # Detect which active points escaped in this iteration
        # |z_n| > escape_radius
        escaped_now = np.abs(z[active]) > escape_radius
        
        # Map back to the full grid layout
        escaped_full = np.zeros_like(active)
        escaped_full[active] = escaped_now

        # Record escape iteration (1-indexed for clarity)
        escape_times[escaped_full] = i + 1

        # Remove escaped points from the active mask
        active[escaped_full] = False

    # 5. Classify final states
    unresolved_mask = active.copy()
    escaped_mask = ~active

    return escape_times, escaped_mask, unresolved_mask, c_matrix

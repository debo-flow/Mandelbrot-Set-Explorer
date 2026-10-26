"""
Julia Set Computation Engine.
Handles the mathematical generation of filled Julia sets for the quadratic family z_{n+1} = z_n^2 + c.
"""
import math
import numpy as np
from typing import Tuple, Dict, Any

# Memory Safeguard: Prevent arrays larger than ~64 million pixels
MAX_SAFE_PIXELS = 64_000_000

def compute_julia(
    re_min: float = -2.0,
    re_max: float = 2.0,
    im_min: float = -2.0,
    im_max: float = 2.0,
    width: int = 800,
    height: int = 800,
    max_iter: int = 100,
    escape_radius: float = 2.0,
    trap_type: str = "None",
    trap_params: Dict[str, Any] = None,
    c: complex = 0.0 + 0.0j
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the filled Julia set for a fixed parameter c.
    Initializes z_0 as the complex plane grid coordinates.

    Returns:
        escape_times (np.ndarray): 2D integer array (escaped iteration).
        smooth_escape_times (np.ndarray): 2D float array (continuous escape).
        trap_distances (np.ndarray): 2D float array (min distance to orbit trap).
        escaped_mask (np.ndarray): 2D boolean array (True = escaped).
        unresolved_mask (np.ndarray): 2D boolean array (True = unresolved).
        z0_matrix (np.ndarray): 2D complex coordinate grid.
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
        
    if width * height > MAX_SAFE_PIXELS:
        raise ValueError(f"Safety limit exceeded: Requesting {width * height:,} pixels. Max allowed is {MAX_SAFE_PIXELS:,}.")

    trap_params = trap_params or {}

    # 2. Coordinate Grid Generation (Initial z_0)
    x = np.linspace(re_min, re_max, width, dtype=np.float64, endpoint=True)
    y = np.linspace(im_min, im_max, height, dtype=np.float64, endpoint=True)
    X, Y = np.meshgrid(x, y)
    z = X + 1j * Y
    z0_matrix = z.copy()

    # 3. Initialization
    escape_times = np.zeros(z.shape, dtype=np.int32)
    smooth_escape_times = np.zeros(z.shape, dtype=np.float64)
    
    if trap_type != "None":
        trap_distances = np.full(z.shape, np.inf, dtype=np.float64)
    else:
        trap_distances = np.zeros(z.shape, dtype=np.float64)
    
    active = np.ones(z.shape, dtype=bool)

    # 4. Escape-time & Orbit-Trap Iteration
    for i in range(max_iter):
        if not np.any(active):
            break

        # Calculate Orbit Trap Distance before squaring
        if trap_type != "None":
            if trap_type == "Point":
                p = trap_params.get('p', 0j)
                d = np.abs(z[active] - p)
            elif trap_type == "Circle":
                p = trap_params.get('p', 0j)
                r = max(trap_params.get('r', 1.0), 1e-9)
                d = np.abs(np.abs(z[active] - p) - r)
            elif trap_type == "Line":
                A = trap_params.get('A', 1.0)
                B = trap_params.get('B', 0.0)
                C = trap_params.get('C', 0.0)
                den = math.hypot(A, B)
                if den == 0: den = 1.0
                d = np.abs(A * z[active].real + B * z[active].imag + C) / den
            elif trap_type == "Cross":
                cx = trap_params.get('x', 0.0)
                cy = trap_params.get('y', 0.0)
                d = np.minimum(np.abs(z[active].real - cx), np.abs(z[active].imag - cy))
            else:
                d = np.inf
                
            trap_distances[active] = np.minimum(trap_distances[active], d)

        # Iterate Z
        z[active] = z[active]**2 + c
        escaped_now = np.abs(z[active]) > escape_radius
        
        if np.any(escaped_now):
            escaped_full = np.zeros_like(active)
            escaped_full[active] = escaped_now

            escape_times[escaped_full] = i + 1

            z_esc = np.abs(z[escaped_full])
            log_z = np.log(z_esc)
            smooth_escape_times[escaped_full] = (i + 1) - np.log2(log_z)

            active[escaped_full] = False

    # 5. Classify final states
    unresolved_mask = active.copy()
    escaped_mask = ~active

    return escape_times, smooth_escape_times, trap_distances, escaped_mask, unresolved_mask, z0_matrix

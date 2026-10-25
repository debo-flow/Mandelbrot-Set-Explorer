import pytest
import numpy as np
from mandelbrot import compute_mandelbrot, MAX_SAFE_PIXELS

def test_mandelbrot_c_0():
    e_times, s_times, traps, escaped, unresolved, c_matrix = compute_mandelbrot(
        re_min=-0.1, re_max=0.1, im_min=-0.1, im_max=0.1, width=3, height=3, max_iter=50
    )
    assert np.isclose(c_matrix[1, 1], 0j)
    assert unresolved[1, 1] == True
    assert e_times[1, 1] == 0
    # Trap defaults to 0 when disabled
    assert traps[1, 1] == 0.0

def test_mandelbrot_orbit_trap_tracking():
    """Validates that Point trap tracks the distance correctly."""
    _, _, traps, _, _, c_mat = compute_mandelbrot(
        re_min=0.0, re_max=0.0, im_min=0.0, im_max=0.0, width=1, height=1, max_iter=10,
        trap_type="Point", trap_params={'p': 1.0 + 0j}
    )
    # z_0 = 0. Distance to p(1.0) is 1.0.
    assert np.isclose(traps[0, 0], 1.0)

def test_safety_limit_rejection():
    with pytest.raises(ValueError, match="Safety limit exceeded"):
        compute_mandelbrot(width=10000, height=10000)

def test_array_validation():
    width, height = 10, 15
    e_times, s_times, traps, escaped, unresolved, c_mat = compute_mandelbrot(
        width=width, height=height
    )
    assert e_times.shape == (height, width)
    assert s_times.shape == (height, width)
    assert traps.shape == (height, width)
    assert escaped.shape == (height, width)
    assert unresolved.shape == (height, width)
    
    assert escaped.dtype == bool
    assert unresolved.dtype == bool
    assert e_times.dtype == np.int32
    assert s_times.dtype == np.float64
    assert traps.dtype == np.float64

def test_invalid_inputs():
    with pytest.raises(ValueError): compute_mandelbrot(re_min=1.0, re_max=-1.0)
    with pytest.raises(ValueError): compute_mandelbrot(re_min=1.0, re_max=1.0)
    with pytest.raises(ValueError): compute_mandelbrot(width=0, height=-10)
    with pytest.raises(ValueError): compute_mandelbrot(max_iter=0)
    with pytest.raises(ValueError): compute_mandelbrot(re_min=float('nan'))
    with pytest.raises(ValueError): compute_mandelbrot(re_max=float('inf'))

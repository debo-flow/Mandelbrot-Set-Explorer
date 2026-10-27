import pytest
import numpy as np
from julia import compute_julia, MAX_SAFE_PIXELS

def test_julia_c_0():
    """Mathematical Correctness: For c=0, z0=0 remains bounded (unresolved)."""
    e_times, s_times, traps, escaped, unresolved, z_matrix = compute_julia(
        re_min=-0.1, re_max=0.1, im_min=-0.1, im_max=0.1, width=3, height=3, max_iter=50, c=0j
    )
    # Center point is exactly z0 = 0
    assert np.isclose(z_matrix[1, 1], 0j)
    assert unresolved[1, 1] == True
    assert e_times[1, 1] == 0

def test_julia_escape_c_minus_1():
    """Mathematical Correctness: For c=-1, z0=2 escapes quickly."""
    e_times, s_times, traps, escaped, unresolved, z_mat = compute_julia(
        re_min=1.9, re_max=2.1, im_min=-0.1, im_max=0.1, width=3, height=3, max_iter=10, c=-1.0+0j
    )
    # Center is z0=2
    assert escaped[1, 1] == True
    # z0=2. z1 = 4 - 1 = 3 (> 2). Escapes at iteration 1.
    assert e_times[1, 1] == 1

def test_safety_limit_rejection():
    with pytest.raises(ValueError, match="Safety limit exceeded"):
        compute_julia(width=10000, height=10000)

def test_array_validation():
    width, height = 10, 15
    e_times, s_times, traps, escaped, unresolved, z_mat = compute_julia(
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
    with pytest.raises(ValueError): compute_julia(re_min=1.0, re_max=-1.0)
    with pytest.raises(ValueError): compute_julia(re_min=1.0, re_max=1.0)
    with pytest.raises(ValueError): compute_julia(width=0, height=-10)
    with pytest.raises(ValueError): compute_julia(max_iter=0)
    with pytest.raises(ValueError): compute_julia(re_min=float('nan'))

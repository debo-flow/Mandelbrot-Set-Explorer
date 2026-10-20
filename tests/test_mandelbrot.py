import pytest
import numpy as np
from mandelbrot import compute_mandelbrot, MAX_SAFE_PIXELS

def test_mandelbrot_c_0():
    e_times, s_times, escaped, unresolved, c_matrix = compute_mandelbrot(
        re_min=-0.1, re_max=0.1, im_min=-0.1, im_max=0.1, width=3, height=3, max_iter=50
    )
    assert np.isclose(c_matrix[1, 1], 0j)
    assert unresolved[1, 1] == True
    assert e_times[1, 1] == 0

def test_safety_limit_rejection():
    """Validates that massive supersampled dimensions are safely rejected."""
    # Force a failure (e.g. 10000 x 10000 = 100,000,000 pixels)
    with pytest.raises(ValueError, match="Safety limit exceeded"):
        compute_mandelbrot(width=10000, height=10000)

def test_array_validation():
    width, height = 10, 15
    e_times, s_times, escaped, unresolved, c_mat = compute_mandelbrot(
        width=width, height=height
    )
    
    assert e_times.shape == (height, width)
    assert s_times.shape == (height, width)
    assert escaped.shape == (height, width)
    assert unresolved.shape == (height, width)
    assert c_mat.shape == (height, width)
    
    assert escaped.dtype == bool
    assert unresolved.dtype == bool
    assert e_times.dtype == np.int32
    assert s_times.dtype == np.float64
    assert c_mat.dtype == np.complex128

def test_invalid_inputs():
    with pytest.raises(ValueError): compute_mandelbrot(re_min=1.0, re_max=-1.0)
    with pytest.raises(ValueError): compute_mandelbrot(re_min=1.0, re_max=1.0)
    with pytest.raises(ValueError): compute_mandelbrot(width=0, height=-10)
    with pytest.raises(ValueError): compute_mandelbrot(max_iter=0)
    with pytest.raises(ValueError): compute_mandelbrot(re_min=float('nan'))
    with pytest.raises(ValueError): compute_mandelbrot(re_max=float('inf'))

import pytest
import numpy as np
from mandelbrot import compute_mandelbrot

def test_mandelbrot_c_0():
    """Mathematical Correctness: c=0 remains bounded (unresolved)."""
    e_times, s_times, escaped, unresolved, c_matrix = compute_mandelbrot(
        re_min=-0.1, re_max=0.1, im_min=-0.1, im_max=0.1, width=3, height=3, max_iter=50
    )
    assert np.isclose(c_matrix[1, 1], 0j)
    assert unresolved[1, 1] == True
    assert escaped[1, 1] == False
    assert e_times[1, 1] == 0

def test_mandelbrot_c_1_smooth():
    """Mathematical Correctness: Smooth coloring handles escaped point."""
    e_times, s_times, escaped, unresolved, c_mat = compute_mandelbrot(
        re_min=0.9, re_max=1.1, im_min=-0.1, im_max=0.1, width=3, height=3, max_iter=10
    )
    # Center c=1
    assert escaped[1, 1] == True
    assert e_times[1, 1] == 3
    
    # Validate Smooth Math: Must be finite and non-zero
    assert np.isfinite(s_times[1, 1])
    assert s_times[1, 1] > 0.0

def test_mandelbrot_c_2():
    """Mathematical Correctness: c=2 escapes."""
    e_times, s_times, escaped, unresolved, c_matrix = compute_mandelbrot(
        re_min=1.9, re_max=2.1, im_min=-0.1, im_max=0.1, width=3, height=3, max_iter=10
    )
    assert escaped[1, 1] == True
    assert e_times[1, 1] == 2

def test_unresolved_smooth_is_zero():
    """Ensure interior pixels don't receive invalid smooth values."""
    e_times, s_times, escaped, unresolved, c_mat = compute_mandelbrot(
        re_min=-0.1, re_max=0.1, im_min=-0.1, im_max=0.1, width=3, height=3, max_iter=50
    )
    # Center c=0
    assert unresolved[1, 1] == True
    assert s_times[1, 1] == 0.0  # Should be untouched by smooth math

def test_array_validation():
    """Array Shape Validation updated for M3."""
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
    """Input Validation."""
    with pytest.raises(ValueError): # Reversed Real boundaries
        compute_mandelbrot(re_min=1.0, re_max=-1.0)
    with pytest.raises(ValueError): # Zero-width
        compute_mandelbrot(re_min=1.0, re_max=1.0)
    with pytest.raises(ValueError): # Invalid dimensions
        compute_mandelbrot(width=0, height=-10)
    with pytest.raises(ValueError): # Invalid iterations
        compute_mandelbrot(max_iter=0)
    with pytest.raises(ValueError): # NaN bounds
        compute_mandelbrot(re_min=float('nan'))
    with pytest.raises(ValueError): # Inf bounds
        compute_mandelbrot(re_max=float('inf'))

import pytest
import numpy as np
from mandelbrot import compute_mandelbrot

def test_mandelbrot_c_0():
    """Mathematical Correctness: c=0 remains bounded (unresolved)."""
    # Create a 3x3 grid centered exactly at 0,0
    escape_times, escaped, unresolved, c_matrix = compute_mandelbrot(
        re_min=-0.1, re_max=0.1, im_min=-0.1, im_max=0.1, width=3, height=3, max_iter=50
    )
    # The center point [1, 1] is exactly c=0
    assert np.isclose(c_matrix[1, 1], 0j)
    assert unresolved[1, 1] == True
    assert escaped[1, 1] == False
    assert escape_times[1, 1] == 0

def test_mandelbrot_c_1():
    """Mathematical Correctness: c=1 escapes quickly."""
    escape_times, escaped, unresolved, c_matrix = compute_mandelbrot(
        re_min=0.9, re_max=1.1, im_min=-0.1, im_max=0.1, width=3, height=3, max_iter=10
    )
    # Center is c=1
    assert np.isclose(c_matrix[1, 1].real, 1.0)
    assert escaped[1, 1] == True
    assert unresolved[1, 1] == False
    # z_0=0, z_1=1, z_2=2, z_3=5 (>2). Escapes at iteration 3
    assert escape_times[1, 1] == 3

def test_mandelbrot_c_2():
    """Mathematical Correctness: c=2 escapes."""
    escape_times, escaped, unresolved, c_matrix = compute_mandelbrot(
        re_min=1.9, re_max=2.1, im_min=-0.1, im_max=0.1, width=3, height=3, max_iter=10
    )
    assert escaped[1, 1] == True
    # z_0=0, z_1=2, z_2=6 (>2). Escapes at iteration 2
    assert escape_times[1, 1] == 2

def test_array_validation():
    """Array Shape and DataType validation."""
    width, height = 10, 15
    e_times, escaped, unresolved, c_mat = compute_mandelbrot(
        width=width, height=height
    )
    
    assert e_times.shape == (height, width)
    assert escaped.shape == (height, width)
    assert unresolved.shape == (height, width)
    assert c_mat.shape == (height, width)
    
    assert escaped.dtype == bool
    assert unresolved.dtype == bool
    assert e_times.dtype == np.int32
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

import pytest
import numpy as np
import matplotlib.pyplot as plt
from visualization import create_mandelbrot_figure, figure_to_png_buffer

@pytest.fixture
def dummy_data():
    escape_times = np.array([[1, 2], [3, 0]], dtype=np.int32)
    unresolved_mask = np.array([[False, False], [False, True]])
    return escape_times, unresolved_mask

def test_figure_generation_and_export(dummy_data):
    """Visualization: Validates generation, PNG export, and figure closure."""
    e_times, unresolved = dummy_data
    fig = create_mandelbrot_figure(
        e_times, unresolved, re_min=-1, re_max=1, im_min=-1, im_max=1
    )
    
    assert isinstance(fig, plt.Figure)
    
    # Test PNG Buffer
    buf = figure_to_png_buffer(fig)
    buf_bytes = buf.read()
    
    # Check PNG magic signature (8 bytes: 89 50 4E 47 0D 0A 1A 0A)
    assert buf_bytes.startswith(b'\x89PNG')
    
    # Clean up to prevent memory leak
    plt.close(fig)

import pytest
import numpy as np
import matplotlib.pyplot as plt
from visualization import create_mandelbrot_figure, figure_to_png_buffer

@pytest.fixture
def dummy_data():
    e_times = np.array([[1, 2], [3, 0]], dtype=np.int32)
    s_times = np.array([[1.5, 2.2], [3.8, 0.0]], dtype=np.float64)
    unresolved_mask = np.array([[False, False], [False, True]])
    return e_times, s_times, unresolved_mask

@pytest.mark.parametrize("mode", ["Classic", "Smooth", "Binary", "Grayscale", "Logarithmic"])
def test_figure_modes(dummy_data, mode):
    """Tests all visualization modes compile successfully without throwing Matplotlib errors."""
    e_times, s_times, unresolved = dummy_data
    fig = create_mandelbrot_figure(
        e_times, s_times, unresolved,
        re_min=-1, re_max=1, im_min=-1, im_max=1,
        mode=mode, cmap_name='viridis', interior_color='black'
    )
    
    assert isinstance(fig, plt.Figure)
    
    buf = figure_to_png_buffer(fig)
    assert buf.read().startswith(b'\x89PNG')
    plt.close(fig)

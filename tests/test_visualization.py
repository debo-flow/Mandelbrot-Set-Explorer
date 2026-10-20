import pytest
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from visualization import colorize_fractal, downsample_image, create_mandelbrot_figure, figure_to_png_buffer

@pytest.fixture
def dummy_data():
    e_times = np.array([[1, 2], [3, 0]], dtype=np.int32)
    s_times = np.array([[1.5, 2.2], [3.8, 0.0]], dtype=np.float64)
    unresolved_mask = np.array([[False, False], [False, True]])
    return e_times, s_times, unresolved_mask

@pytest.mark.parametrize("mode", ["Classic", "Smooth", "Binary", "Grayscale", "Logarithmic"])
def test_colorize_fractal_pipeline(dummy_data, mode):
    """Tests the NumPy-to-Pillow color mapping architecture."""
    e_times, s_times, unresolved = dummy_data
    
    img, sm = colorize_fractal(e_times, s_times, unresolved, mode, 'viridis', 'black')
    
    assert isinstance(img, Image.Image)
    assert img.width == 2
    assert img.height == 2
    assert img.mode == 'RGB'

def test_downsample_image():
    """Ensures supersampling filters reduce pixel grids correctly."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    
    # Simulate a 2x downsample back to 50x50
    downsampled = downsample_image(img, 50, 50, 'LANCZOS')
    
    assert downsampled.width == 50
    assert downsampled.height == 50

def test_figure_metadata_export(dummy_data):
    """Tests embedding PNG metadata."""
    e_times, s_times, unresolved = dummy_data
    img, sm = colorize_fractal(e_times, s_times, unresolved, 'Smooth', 'viridis', 'black')
    
    fig = create_mandelbrot_figure(img, sm, -1, 1, -1, 1, 'Smooth')
    
    metadata = {"TestKey": "TestValue"}
    buf = figure_to_png_buffer(fig, metadata)
    
    assert buf.read().startswith(b'\x89PNG')
    plt.close(fig)

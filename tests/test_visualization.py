import pytest
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from visualization import colorize_fractal, downsample_image, create_fractal_figure, figure_to_png_buffer

@pytest.fixture
def dummy_data():
    e_times = np.array([[1, 2], [3, 0]], dtype=np.int32)
    s_times = np.array([[1.5, 2.2], [3.8, 0.0]], dtype=np.float64)
    traps = np.array([[0.5, 0.1], [1.2, np.inf]], dtype=np.float64)
    unresolved_mask = np.array([[False, False], [False, True]])
    return e_times, s_times, traps, unresolved_mask

@pytest.mark.parametrize("mode", ["Classic", "Smooth", "Orbit Trap", "Orbit Trap Blend", "Logarithmic"])
def test_colorize_fractal_pipeline(dummy_data, mode):
    e_times, s_times, traps, unresolved = dummy_data
    img, sm = colorize_fractal(e_times, s_times, traps, unresolved, mode, 'viridis', 'black')
    assert isinstance(img, Image.Image)
    assert img.width == 2

def test_downsample_image():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    downsampled = downsample_image(img, 50, 50, 'LANCZOS')
    assert downsampled.width == 50

def test_figure_metadata_export(dummy_data):
    e_times, s_times, traps, unresolved = dummy_data
    img, sm = colorize_fractal(e_times, s_times, traps, unresolved, 'Smooth', 'viridis', 'black')
    
    fig = create_fractal_figure(img, sm, -1, 1, -1, 1, 'Smooth', marker_c=0j)
    metadata = {"TestKey": "TestValue"}
    buf = figure_to_png_buffer(fig, metadata)
    
    assert buf.read().startswith(b'\x89PNG')
    plt.close(fig)

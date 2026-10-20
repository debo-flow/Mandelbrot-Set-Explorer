"""
Scientific Visualization module for the Mandelbrot Set.
Handles mathematical color mapping independent of numerical iteration.
"""
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_mandelbrot_figure(
    escape_times: np.ndarray,
    smooth_escape_times: np.ndarray,
    unresolved_mask: np.ndarray,
    re_min: float,
    re_max: float,
    im_min: float,
    im_max: float,
    mode: str = 'Smooth',
    cmap_name: str = 'magma',
    interior_color: str = 'black'
) -> plt.Figure:
    """
    Renders Mandelbrot Set arrays into a scientifically accurate Matplotlib Figure.
    """
    aspect_ratio = (im_max - im_min) / (re_max - re_min)
    base_width = 10.0
    fig, ax = plt.subplots(figsize=(base_width, base_width * aspect_ratio))
    
    # Mode selection logic
    if mode == 'Classic':
        data = escape_times.astype(float)
        cmap = plt.get_cmap(cmap_name).copy()
    elif mode == 'Smooth':
        data = smooth_escape_times
        cmap = plt.get_cmap(cmap_name).copy()
    elif mode == 'Binary':
        data = np.ones_like(escape_times, dtype=float)
        cmap = mcolors.ListedColormap(['white'])
    elif mode == 'Grayscale':
        data = smooth_escape_times
        cmap = plt.get_cmap('gray').copy()
    elif mode == 'Logarithmic':
        # Safely log-scale the positive smooth data
        data = np.log1p(np.maximum(0, smooth_escape_times))
        cmap = plt.get_cmap(cmap_name).copy()
    else:
        data = smooth_escape_times
        cmap = plt.get_cmap(cmap_name).copy()

    # Apply the mask to explicitly isolate unresolved/interior points
    masked_data = np.ma.masked_where(unresolved_mask, data)
    cmap.set_bad(color=interior_color)
    
    # Plot using imshow
    im = ax.imshow(
        masked_data, 
        cmap=cmap, 
        origin='lower',
        extent=(re_min, re_max, im_min, im_max),
        interpolation='nearest'
    )
    
    ax.set_xlabel("Real Axis ($\Re$)")
    ax.set_ylabel("Imaginary Axis ($\Im$)")
    ax.set_title(f"Mandelbrot Set: {mode} Mode")
    
    if mode != 'Binary':
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Escape Metric")
    
    fig.tight_layout()
    return fig

def figure_to_png_buffer(fig: plt.Figure) -> io.BytesIO:
    """Safely exports a Matplotlib figure to an in-memory PNG buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

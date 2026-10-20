"""
Scientific Visualization module for the Mandelbrot Set.
"""
import io
import numpy as np
import matplotlib.pyplot as plt

def create_mandelbrot_figure(
    escape_times: np.ndarray,
    unresolved_mask: np.ndarray,
    re_min: float,
    re_max: float,
    im_min: float,
    im_max: float,
    cmap_name: str = 'magma'
) -> plt.Figure:
    """
    Renders the computed Mandelbrot Set arrays into a scientifically 
    accurate Matplotlib Figure.
    """
    # Calculate aspect ratio to prevent image distortion
    aspect_ratio = (im_max - im_min) / (re_max - re_min)
    base_width = 10.0
    
    fig, ax = plt.subplots(figsize=(base_width, base_width * aspect_ratio))
    
    # Mask the escape_times array so unresolved points can be colored independently
    masked_escape_times = np.ma.masked_where(unresolved_mask, escape_times)
    
    # Configure the colormap
    try:
        cmap = plt.get_cmap(cmap_name).copy()
    except ValueError:
        cmap = plt.get_cmap('magma').copy()
        
    # Unresolved points are rendered explicitly as black
    cmap.set_bad(color='black')
    
    # Plot using imshow
    im = ax.imshow(
        masked_escape_times, 
        cmap=cmap, 
        origin='lower',  # ensures y-axis increases upwards
        extent=(re_min, re_max, im_min, im_max),
        interpolation='nearest'
    )
    
    ax.set_xlabel("Real Axis ($\Re$)")
    ax.set_ylabel("Imaginary Axis ($\Im$)")
    ax.set_title(f"Mandelbrot Set ($z_{{n+1}} = z_n^2 + c$)")
    
    # Add Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Escape Iteration Count")
    
    fig.tight_layout()
    return fig

def figure_to_png_buffer(fig: plt.Figure) -> io.BytesIO:
    """
    Safely exports a Matplotlib figure to an in-memory PNG buffer.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

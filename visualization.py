"""
Scientific Visualization module for the Mandelbrot Set.
Handles mathematical color mapping, Pillow-based supersampling filters, and rendering.
"""
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from PIL import Image, PngImagePlugin
from typing import Tuple

def colorize_fractal(
    escape_times: np.ndarray,
    smooth_escape_times: np.ndarray,
    unresolved_mask: np.ndarray,
    mode: str,
    cmap_name: str,
    interior_color: str
) -> Tuple[Image.Image, cm.ScalarMappable]:
    """Applies colormaps at the raw mathematical resolution to produce a PIL Image."""
    
    # 1. Mode Transformation
    if mode == 'Classic':
        data = escape_times.astype(float)
    elif mode == 'Logarithmic':
        data = np.log1p(np.maximum(0, smooth_escape_times))
    elif mode == 'Binary':
        data = np.ones_like(escape_times, dtype=float)
        cmap_name = 'gray'
    elif mode == 'Grayscale':
        data = smooth_escape_times
        cmap_name = 'gray'
    else: # Smooth
        data = smooth_escape_times
        
    # 2. Normalize and Setup Colormap
    if mode == 'Binary':
        cmap = mcolors.ListedColormap(['white'])
        norm = mcolors.Normalize(vmin=0, vmax=1)
    else:
        try:
            cmap = plt.get_cmap(cmap_name).copy()
        except ValueError:
            cmap = plt.get_cmap('magma').copy()
            
        valid_data = data[~unresolved_mask]
        if len(valid_data) > 0:
            norm = mcolors.Normalize(vmin=valid_data.min(), vmax=valid_data.max())
        else:
            norm = mcolors.Normalize(vmin=0, vmax=1)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    
    # 3. Apply Colors (Yields an RGBA float array [0.0 - 1.0])
    rgba = sm.to_rgba(data)
    
    # 4. Handle Unresolved Interior
    try:
        ic = mcolors.to_rgba(interior_color)
    except ValueError:
        ic = (0.0, 0.0, 0.0, 1.0)
    rgba[unresolved_mask] = ic
    
    # 5. Convert to uint8 RGB Image
    rgb8 = (rgba[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb8)
    
    return img, sm

def downsample_image(img: Image.Image, target_width: int, target_height: int, filter_name: str) -> Image.Image:
    """Downsamples a high-resolution PIL Image using scientific anti-aliasing filters."""
    if img.width == target_width and img.height == target_height:
        return img # No resampling needed if factor is 1x
        
    filters = {
        'LANCZOS': Image.Resampling.LANCZOS,
        'BICUBIC': Image.Resampling.BICUBIC,
        'BILINEAR': Image.Resampling.BILINEAR,
        'BOX': Image.Resampling.BOX
    }
    
    resample_filter = filters.get(filter_name.upper(), Image.Resampling.LANCZOS)
    return img.resize((target_width, target_height), resample=resample_filter)

def create_mandelbrot_figure(
    img: Image.Image,
    sm: cm.ScalarMappable,
    re_min: float,
    re_max: float,
    im_min: float,
    im_max: float,
    mode: str
) -> plt.Figure:
    """Wraps the resampled Pillow image into a scientific Matplotlib figure."""
    aspect_ratio = img.height / img.width
    base_width = 10.0
    fig, ax = plt.subplots(figsize=(base_width, base_width * aspect_ratio))
    
    # Since y arrays go from min to max, row 0 is bottom. PIL displays row 0 as top.
    # extent=(left, right, bottom, top) combined with origin='lower' inherently maps correctly.
    ax.imshow(
        np.array(img),
        origin='lower',
        extent=(re_min, re_max, im_min, im_max),
        interpolation='none' # Pre-interpolated natively by Pillow!
    )
    
    ax.set_xlabel("Real Axis ($\Re$)")
    ax.set_ylabel("Imaginary Axis ($\Im$)")
    ax.set_title(f"Mandelbrot Set: {mode} Mode")
    
    if mode != 'Binary':
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Escape Metric")
    
    fig.tight_layout()
    return fig

def figure_to_png_buffer(fig: plt.Figure, metadata: dict = None) -> io.BytesIO:
    """Safely exports a figure to memory, optionally injecting PNG Metadata."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    if metadata:
        img = Image.open(buf)
        meta = PngImagePlugin.PngInfo()
        for k, v in metadata.items():
            meta.add_text(k, str(v))
        
        out_buf = io.BytesIO()
        img.save(out_buf, format='png', pnginfo=meta)
        out_buf.seek(0)
        return out_buf
        
    return buf

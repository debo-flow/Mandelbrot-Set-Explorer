"""
Streamlit Application UI for the Mandelbrot Set Explorer v4.0.0.
"""
import time
import streamlit as st
import matplotlib.pyplot as plt
from mandelbrot import compute_mandelbrot, MAX_SAFE_PIXELS
from visualization import colorize_fractal, downsample_image, create_mandelbrot_figure, figure_to_png_buffer
from navigation import Viewport, HistoryManager

st.set_page_config(page_title="Mandelbrot Explorer v4.0", page_icon="🌌", layout="wide")

DEFAULT_VIEW = Viewport(-2.5, 1.0, -1.5, 1.5)

if 'history' not in st.session_state:
    st.session_state.history = HistoryManager()
    st.session_state.history.push(DEFAULT_VIEW)
if 'pan_step' not in st.session_state:
    st.session_state.pan_step = 0.20
if 'zoom_factor' not in st.session_state:
    st.session_state.zoom_factor = 2.0

# Critical: Caching restricted strictly to the numerical math engine
@st.cache_data(show_spinner=False)
def cached_compute(re_min, re_max, im_min, im_max, w, h, iters, radius):
    return compute_mandelbrot(re_min, re_max, im_min, im_max, w, h, iters, radius)

def navigate(action: str, **kwargs):
    current_vp = st.session_state.history.current
    try:
        if action == "zoom_in":
            new_vp = current_vp.zoom(st.session_state.zoom_factor)
        elif action == "zoom_out":
            new_vp = current_vp.zoom(1.0 / st.session_state.zoom_factor)
        elif action == "pan_up":
            new_vp = current_vp.pan(0, st.session_state.pan_step)
        elif action == "pan_down":
            new_vp = current_vp.pan(0, -st.session_state.pan_step)
        elif action == "pan_left":
            new_vp = current_vp.pan(-st.session_state.pan_step, 0)
        elif action == "pan_right":
            new_vp = current_vp.pan(st.session_state.pan_step, 0)
        elif action == "custom":
            cx, cy, span = kwargs['cx'], kwargs['cy'], kwargs['span']
            new_vp = Viewport(cx - span/2, cx + span/2, cy - span/2, cy + span/2)
        elif action == "reset":
            new_vp = DEFAULT_VIEW
        elif action == "undo":
            st.session_state.history.undo()
            return
        elif action == "redo":
            st.session_state.history.redo()
            return
        else:
            return
        st.session_state.history.push(new_vp)
    except ValueError as e:
        st.error(f"Navigation Error: {e}")

def main():
    st.title("High-Resolution Supersampling & Anti-Aliasing Laboratory")
    
    # 1. Workflow Selection
    st.sidebar.header("1. Workflow")
    workflow = st.sidebar.radio("Mode", ["Interactive Preview", "Final Render"])

    # 2. Base Output Dimensions
    st.sidebar.header("2. Rendering Resolution")
    preset = st.sidebar.selectbox("Quality Preset", ["STANDARD", "HIGH QUALITY", "CUSTOM"])
    
    if workflow == "Interactive Preview":
        out_w, out_h = 600, 600
        st.sidebar.caption("Preview mode forces low resolutions to prioritize navigation speed.")
    else:
        if preset == "STANDARD":
            w_def, h_def = 800, 800
            disabled = True
        elif preset == "HIGH QUALITY":
            w_def, h_def = 1600, 1600
            disabled = True
        else:
            w_def, h_def = 800, 800
            disabled = False
            
        out_w = st.sidebar.number_input("Output Width (px)", value=w_def, disabled=disabled, step=100, min_value=100)
        out_h = st.sidebar.number_input("Output Height (px)", value=h_def, disabled=disabled, step=100, min_value=100)

    # 3. Supersampling & Math Depth
    st.sidebar.header("3. Supersampling & Math")
    if workflow == "Interactive Preview":
        ss_factor = 1
        resample_filter = "BOX"
        max_iter = st.sidebar.number_input("Max Iterations", 10, 2000, 50, step=10)
        st.sidebar.caption("Supersampling is disabled during Preview.")
    else:
        ss_factor = st.sidebar.selectbox("Internal Supersampling Factor", [1, 2, 3, 4], index=1)
        resample_filter = st.sidebar.selectbox("Downsampling Filter", ["LANCZOS", "BICUBIC", "BILINEAR", "BOX"])
        max_iter = st.sidebar.number_input("Max Iterations", 10, 5000, 150, step=10)

    # 4. Visualization Settings
    st.sidebar.header("4. Visualization Settings")
    mode = st.sidebar.selectbox("Visualization Mode", ["Smooth", "Classic", "Binary", "Grayscale", "Logarithmic"])
    cmap_list = ["magma", "inferno", "plasma", "viridis", "turbo", "twilight", "hot", "cool", "hsv"]
    cmap = st.sidebar.selectbox("Color Map", cmap_list, disabled=(mode in ["Binary", "Grayscale"]))
    interior_color = st.sidebar.selectbox("Interior (Unresolved) Color", ["black", "white", "darkblue", "darkred"])

    # Aspect Ratio & Memory Validation
    raw_vp = st.session_state.history.current
    vp = raw_vp.enforce_aspect_ratio(out_w, out_h)
    cx, cy = vp.center
    
    internal_w = out_w * ss_factor
    internal_h = out_h * ss_factor
    internal_pixels = internal_w * internal_h

    # Top Interface Panels
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 2])
    with nav_col1:
        st.markdown("**Zoom Controls**")
        st.session_state.zoom_factor = st.number_input("Zoom Factor", value=2.0, min_value=1.1, step=0.5)
        cols = st.columns(2)
        cols[0].button("🔍 In", on_click=navigate, args=("zoom_in",), use_container_width=True)
        cols[1].button("🔎 Out", on_click=navigate, args=("zoom_out",), use_container_width=True)
    with nav_col2:
        st.markdown("**Pan Controls**")
        st.session_state.pan_step = st.number_input("Pan Step (%)", value=0.20, min_value=0.01, step=0.05)
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col2.button("⬆️", on_click=navigate, args=("pan_up",), use_container_width=True)
        p_col1.button("⬅️", on_click=navigate, args=("pan_left",), use_container_width=True)
        p_col3.button("➡️", on_click=navigate, args=("pan_right",), use_container_width=True)
        p_col2.button("⬇️", on_click=navigate, args=("pan_down",), use_container_width=True)
    with nav_col3:
        st.markdown("**History & Coordinates**")
        h_col1, h_col2, h_col3 = st.columns(3)
        h_col1.button("↩️ Prev", on_click=navigate, args=("undo",), use_container_width=True)
        h_col2.button("🔄 Reset", on_click=navigate, args=("reset",), use_container_width=True)
        h_col3.button("↪️ Next", on_click=navigate, args=("redo",), use_container_width=True)
        with st.expander("Go to Custom Coordinates"):
            c_re = st.number_input("Real Center", value=float(cx), format="%.6f")
            c_im = st.number_input("Imag Center", value=float(cy), format="%.6f")
            c_span = st.number_input("Span", value=float(vp.width), format="%.6f")
            st.button("Go", on_click=navigate, args=("custom",), kwargs={"cx": c_re, "cy": c_im, "span": c_span})

    st.info(f"**Output Target:** {out_w} x {out_h} | **Internal Sampling:** {internal_w} x {internal_h} ({ss_factor}x) | **Total Pixels:** {internal_pixels:,}")
    
    if internal_pixels > MAX_SAFE_PIXELS:
        st.error(f"Internal pixel count ({internal_pixels:,}) exceeds the safety limit ({MAX_SAFE_PIXELS:,}). Please reduce Output Dimensions or Supersampling.")
        st.stop()

    # Rendering Pipeline
    try:
        t0 = time.perf_counter()
        
        # 1. Numerical Engine (Cached)
        escape_times, smooth_times, escaped, unresolved, _ = cached_compute(
            vp.re_min, vp.re_max, vp.im_min, vp.im_max, internal_w, internal_h, max_iter, 2.0
        )
        t_comp = time.perf_counter()
        
        # 2. Color Mapping
        raw_img, sm = colorize_fractal(escape_times, smooth_times, unresolved, mode, cmap, interior_color)
        t_color = time.perf_counter()
        
        # 3. Downsampling Filter (Anti-Aliasing)
        final_img = downsample_image(raw_img, out_w, out_h, resample_filter)
        t_sample = time.perf_counter()
        
        # 4. Scientific Wrap
        fig = create_mandelbrot_figure(final_img, sm, vp.re_min, vp.re_max, vp.im_min, vp.im_max, mode)
        
        metadata = {
            "Project": "Mandelbrot-Set-Explorer v4.0.0",
            "Mode": mode,
            "Colormap": cmap,
            "Supersampling": f"{ss_factor}x",
            "ResampleFilter": resample_filter,
            "Re_Min": vp.re_min,
            "Re_Max": vp.re_max,
            "Im_Min": vp.im_min,
            "Im_Max": vp.im_max,
            "MaxIter": max_iter
        }
        
        png_buf = figure_to_png_buffer(fig, metadata)
        t_total = time.perf_counter() - t0
        
        st.success(f"Math: {t_comp-t0:.3f}s | Map: {t_color-t_comp:.3f}s | Filter: {t_sample-t_color:.3f}s | Total: {t_total:.3f}s")
        st.pyplot(fig)
        
        st.download_button(
            f"📥 Download High-Res Output (PNG)", 
            data=png_buf, 
            file_name=f"mandelbrot_{ss_factor}x_{out_w}x{out_h}.png", 
            mime="image/png"
        )
        plt.close(fig)
    except Exception as e:
        st.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()

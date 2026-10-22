"""
Streamlit Application UI for the Mandelbrot Set Explorer v5.0.0.
Integrates High-Res Rendering with Deep Zoom Precision Diagnostics.
"""
import time
import streamlit as st
import matplotlib.pyplot as plt
from mandelbrot import compute_mandelbrot, MAX_SAFE_PIXELS
from visualization import colorize_fractal, downsample_image, create_mandelbrot_figure, figure_to_png_buffer
from navigation import Viewport, HistoryManager
from precision import evaluate_precision, PrecisionState

st.set_page_config(page_title="Mandelbrot Explorer v5.0", page_icon="🌌", layout="wide")

DEFAULT_VIEW = Viewport(-2.5, 1.0, -1.5, 1.5)

if 'history' not in st.session_state:
    st.session_state.history = HistoryManager()
    st.session_state.history.push(DEFAULT_VIEW)
if 'pan_step' not in st.session_state:
    st.session_state.pan_step = 0.20

# Strictly cached math engine
@st.cache_data(show_spinner=False)
def cached_compute(re_min, re_max, im_min, im_max, w, h, iters, radius):
    return compute_mandelbrot(re_min, re_max, im_min, im_max, w, h, iters, radius)

def navigate(action: str, **kwargs):
    current_vp = st.session_state.history.current
    try:
        if action == "zoom_in":
            new_vp = current_vp.zoom(kwargs['factor'])
        elif action == "zoom_out":
            new_vp = current_vp.zoom(1.0 / kwargs['factor'])
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
    st.title("Deep Zoom & Numerical Precision Laboratory")
    
    # 1. Output & Rendering Quality
    st.sidebar.header("1. Output Quality")
    workflow = st.sidebar.radio("Mode", ["Interactive Preview", "Final Render"])
    
    if workflow == "Interactive Preview":
        out_w, out_h = 600, 600
        ss_factor = 1
        resample_filter = "BOX"
        max_iter = st.sidebar.number_input("Max Iterations", 10, 2000, 50, step=10)
    else:
        out_w = st.sidebar.number_input("Output Width (px)", value=1200, step=100, min_value=100)
        out_h = st.sidebar.number_input("Output Height (px)", value=800, step=100, min_value=100)
        ss_factor = st.sidebar.selectbox("Supersampling Factor", [1, 2, 3, 4], index=1)
        resample_filter = st.sidebar.selectbox("Downsampling Filter", ["LANCZOS", "BICUBIC", "BILINEAR", "BOX"])
        max_iter = st.sidebar.number_input("Max Iterations", 10, 5000, 250, step=10)

    # 2. Visualization Settings
    st.sidebar.header("2. Visualization Settings")
    mode = st.sidebar.selectbox("Visualization Mode", ["Smooth", "Classic", "Binary", "Grayscale", "Logarithmic"])
    cmap_list = ["magma", "inferno", "plasma", "viridis", "turbo", "twilight", "hot", "cool", "hsv"]
    cmap = st.sidebar.selectbox("Color Map", cmap_list, disabled=(mode in ["Binary", "Grayscale"]))
    interior_color = st.sidebar.selectbox("Interior Color", ["black", "white", "darkblue", "darkred"])

    # Geometry & Aspect Ratio setup
    raw_vp = st.session_state.history.current
    vp = raw_vp.enforce_aspect_ratio(out_w, out_h)
    cx, cy = vp.center
    
    internal_w = out_w * ss_factor
    internal_h = out_h * ss_factor
    internal_pixels = internal_w * internal_h
    zoom_level = DEFAULT_VIEW.width / vp.width

    # TOP MAIN AREA: Precision Diagnostics & Navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1.5, 1, 1])
    
    with nav_col1:
        st.subheader("Deep Zoom Coordinates")
        c_re = st.number_input("Real Center (Re)", value=float(cx), format="%.16f")
        c_im = st.number_input("Imaginary Center (Im)", value=float(cy), format="%.16f")
        c_span = st.number_input("Horizontal Span", value=float(vp.width), format="%.16f")
        if st.button("🚀 Jump to Exact Coordinates"):
            navigate("custom", cx=c_re, cy=c_im, span=c_span)
            st.rerun()

    with nav_col2:
        st.subheader("Zoom & Pan")
        zf = st.number_input("Zoom Factor", value=2.0, min_value=1.1, step=0.5)
        cols = st.columns(2)
        cols[0].button("🔍 In", on_click=navigate, args=("zoom_in",), kwargs={"factor": zf}, use_container_width=True)
        cols[1].button("🔎 Out", on_click=navigate, args=("zoom_out",), kwargs={"factor": zf}, use_container_width=True)
        
        st.session_state.pan_step = st.number_input("Pan Step (%)", value=0.20, min_value=0.01, step=0.05)
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col2.button("⬆️", on_click=navigate, args=("pan_up",), use_container_width=True)
        p_col1.button("⬅️", on_click=navigate, args=("pan_left",), use_container_width=True)
        p_col3.button("➡️", on_click=navigate, args=("pan_right",), use_container_width=True)
        p_col2.button("⬇️", on_click=navigate, args=("pan_down",), use_container_width=True)

    with nav_col3:
        st.subheader("View History")
        st.button("↩️ Previous View", on_click=navigate, args=("undo",), use_container_width=True)
        st.button("↪️ Next View", on_click=navigate, args=("redo",), use_container_width=True)
        st.button("🔄 Reset to Default", on_click=navigate, args=("reset",), use_container_width=True)

    st.markdown("---")
    
    # Evaluate Numerical Precision Constraints BEFORE rendering
    diagnostics = evaluate_precision(vp.re_min, vp.re_max, vp.im_min, vp.im_max, internal_w, internal_h)
    
    # Precision Status Dashboard
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Relative Zoom", f"{zoom_level:,.1e}x")
    metrics_cols[1].metric("Real Span (Width)", f"{vp.width:.2e}")
    metrics_cols[2].metric("Grid Pixel Spacing", f"{diagnostics.horizontal_step:.2e}")
    metrics_cols[3].metric("Float64 Limit", f"{diagnostics.re_spacing:.2e}")
    
    if diagnostics.state == PrecisionState.NORMAL:
        st.success(f"✅ **{diagnostics.state.value}**: {diagnostics.message}")
    elif diagnostics.state == PrecisionState.WARNING:
        st.warning(f"⚠️ **{diagnostics.state.value}**: {diagnostics.message}")
    else:
        st.error(f"🛑 **{diagnostics.state.value}**: {diagnostics.message} Arbitrary precision backend (future work) is required to zoom further.")
    
    if internal_pixels > MAX_SAFE_PIXELS:
        st.error(f"Memory Safety limit exceeded: Requesting {internal_pixels:,} pixels. Max allowed is {MAX_SAFE_PIXELS:,}.")
        st.stop()

    # Rendering Pipeline
    try:
        t0 = time.perf_counter()
        
        escape_times, smooth_times, escaped, unresolved, _ = cached_compute(
            vp.re_min, vp.re_max, vp.im_min, vp.im_max, internal_w, internal_h, max_iter, 2.0
        )
        t_comp = time.perf_counter()
        
        raw_img, sm = colorize_fractal(escape_times, smooth_times, unresolved, mode, cmap, interior_color)
        t_color = time.perf_counter()
        
        final_img = downsample_image(raw_img, out_w, out_h, resample_filter)
        t_sample = time.perf_counter()
        
        fig = create_mandelbrot_figure(final_img, sm, vp.re_min, vp.re_max, vp.im_min, vp.im_max, mode)
        
        metadata = {
            "Project": "Mandelbrot-Set-Explorer v5.0.0",
            "Mode": mode,
            "Colormap": cmap,
            "Supersampling": f"{ss_factor}x",
            "ResampleFilter": resample_filter,
            "Re_Center": str(cx),
            "Im_Center": str(cy),
            "Re_Span": str(vp.width),
            "MaxIter": max_iter,
            "PrecisionState": diagnostics.state.value
        }
        
        png_buf = figure_to_png_buffer(fig, metadata)
        t_total = time.perf_counter() - t0
        
        st.caption(f"⏱️ **Performance:** Math: {t_comp-t0:.3f}s | Color: {t_color-t_comp:.3f}s | Resample: {t_sample-t_color:.3f}s | Total: {t_total:.3f}s")
        st.pyplot(fig)
        
        st.download_button(
            f"📥 Download High-Res Render (PNG)", 
            data=png_buf, 
            file_name=f"mandelbrot_{zoom_level:.1e}x.png", 
            mime="image/png"
        )
        plt.close(fig)
    except Exception as e:
        st.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()

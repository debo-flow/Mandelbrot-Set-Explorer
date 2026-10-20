"""
Streamlit Application UI for the Mandelbrot Set Explorer v3.0.0.
"""
import time
import streamlit as st
import matplotlib.pyplot as plt
from mandelbrot import compute_mandelbrot
from visualization import create_mandelbrot_figure, figure_to_png_buffer
from navigation import Viewport, HistoryManager

st.set_page_config(page_title="Mandelbrot Explorer v3.0", page_icon="🌌", layout="wide")

DEFAULT_VIEW = Viewport(-2.5, 1.0, -1.5, 1.5)
INITIAL_WIDTH = DEFAULT_VIEW.width

if 'history' not in st.session_state:
    st.session_state.history = HistoryManager()
    st.session_state.history.push(DEFAULT_VIEW)
if 'pan_step' not in st.session_state:
    st.session_state.pan_step = 0.20
if 'zoom_factor' not in st.session_state:
    st.session_state.zoom_factor = 2.0

# Critical: Caching is restricted ONLY to the mathematical engine.
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
    st.title("Smooth Dynamics & Advanced Fractal Visualization Laboratory")
    
    # Sidebar: Quality Presets
    st.sidebar.header("1. Rendering Detail")
    preset = st.sidebar.selectbox("Quality Preset", ["LOW", "MEDIUM", "HIGH", "CUSTOM"], index=1)
    
    presets_map = {
        "LOW": (400, 400, 50),
        "MEDIUM": (800, 800, 150),
        "HIGH": (1200, 1200, 500)
    }
    
    if preset != "CUSTOM":
        w_def, h_def, iter_def = presets_map[preset]
        disable_custom = True
    else:
        w_def, h_def, iter_def = 800, 800, 150
        disable_custom = False

    width = st.sidebar.number_input("Width (px)", value=w_def, disabled=disable_custom, step=100)
    height = st.sidebar.number_input("Height (px)", value=h_def, disabled=disable_custom, step=100)
    max_iter = st.sidebar.number_input("Max Iterations", value=iter_def, disabled=disable_custom, step=10)

    # Sidebar: Visualization Settings
    st.sidebar.header("2. Visualization Settings")
    mode = st.sidebar.selectbox("Visualization Mode", ["Smooth", "Classic", "Binary", "Grayscale", "Logarithmic"])
    
    cmap_list = ["magma", "inferno", "plasma", "viridis", "turbo", "twilight", "hot", "cool", "hsv"]
    cmap = st.sidebar.selectbox("Color Map", cmap_list, disabled=(mode in ["Binary", "Grayscale"]))
    interior_color = st.sidebar.selectbox("Interior (Unresolved) Color", ["black", "white", "darkblue", "darkred"])

    raw_vp = st.session_state.history.current
    vp = raw_vp.enforce_aspect_ratio(width, height)
    cx, cy = vp.center
    zoom_level = INITIAL_WIDTH / vp.width

    # Main Area: Navigation Controls
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

    st.caption(f"**Center:** {cx:.6f} + {cy:.6f}i &nbsp;|&nbsp; **Zoom:** {zoom_level:,.2f}x")

    # Computing and Instant Rendering
    try:
        t0 = time.perf_counter()
        
        # 1. Compute Numerical Arrays (Cached operation)
        escape_times, smooth_times, escaped, unresolved, _ = cached_compute(
            vp.re_min, vp.re_max, vp.im_min, vp.im_max, width, height, max_iter, 2.0
        )
        t_comp = time.perf_counter()
        
        # 2. Render Figure (Instant state application)
        fig = create_mandelbrot_figure(
            escape_times, smooth_times, unresolved, 
            vp.re_min, vp.re_max, vp.im_min, vp.im_max, 
            mode, cmap, interior_color
        )
        png_buf = figure_to_png_buffer(fig)
        t_total = time.perf_counter() - t0
        
        st.success(f"Compute: {(t_comp-t0):.3f}s | Total Render time: {t_total:.3f}s. Color updates map instantly.")
        st.pyplot(fig)
        
        st.download_button(
            f"📥 Download {mode} View (PNG)", 
            data=png_buf, 
            file_name=f"mandelbrot_{mode}_{zoom_level:.1f}x.png", 
            mime="image/png"
        )
        plt.close(fig)
    except Exception as e:
        st.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()

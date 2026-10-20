"""
Streamlit Application UI for the Mandelbrot Set Explorer v2.0.0.
"""
import time
import streamlit as st
import matplotlib.pyplot as plt
from mandelbrot import compute_mandelbrot
from visualization import create_mandelbrot_figure, figure_to_png_buffer
from navigation import Viewport, HistoryManager

st.set_page_config(page_title="Mandelbrot Explorer v2.0", page_icon="🌌", layout="wide")

# Constants
DEFAULT_VIEW = Viewport(-2.5, 1.0, -1.5, 1.5)
INITIAL_WIDTH = DEFAULT_VIEW.width

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state.history = HistoryManager()
    st.session_state.history.push(DEFAULT_VIEW)
if 'pan_step' not in st.session_state:
    st.session_state.pan_step = 0.20
if 'zoom_factor' not in st.session_state:
    st.session_state.zoom_factor = 2.0

@st.cache_data(show_spinner=False)
def cached_compute(re_min, re_max, im_min, im_max, w, h, iters, radius):
    return compute_mandelbrot(re_min, re_max, im_min, im_max, w, h, iters, radius)

def navigate(action: str, **kwargs):
    """Handles view transformations and updates history."""
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
            # Jump to custom coordinate with defined width
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
    st.title("Interactive Complex Plane Navigation & Precision Zoom Laboratory")
    
    # 1. Sidebar Configuration
    st.sidebar.header("Resolution & Computation")
    width = st.sidebar.number_input("Image Width (px)", value=800, min_value=100, max_value=2000, step=100)
    height = st.sidebar.number_input("Image Height (px)", value=800, min_value=100, max_value=2000, step=100)
    max_iter = st.sidebar.slider("Max Iterations", 10, 2000, 100, 10)
    cmap = st.sidebar.selectbox("Color Map", ["magma", "inferno", "plasma", "viridis", "cividis", "twilight"])

    # Enforce aspect ratio BEFORE rendering
    raw_vp = st.session_state.history.current
    vp = raw_vp.enforce_aspect_ratio(width, height)
    
    # Mathematical computations for display
    cx, cy = vp.center
    zoom_level = INITIAL_WIDTH / vp.width

    # 2. Main Area: Navigation Controls
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 2])
    
    with nav_col1:
        st.markdown("**Zoom Controls**")
        st.session_state.zoom_factor = st.number_input("Zoom Factor", value=2.0, min_value=1.1, step=0.5)
        cols = st.columns(2)
        cols[0].button("🔍 Zoom In", on_click=navigate, args=("zoom_in",), use_container_width=True)
        cols[1].button("🔎 Zoom Out", on_click=navigate, args=("zoom_out",), use_container_width=True)

    with nav_col2:
        st.markdown("**Pan Controls**")
        st.session_state.pan_step = st.number_input("Pan Step (%)", value=0.20, min_value=0.01, max_value=1.0, step=0.05)
        # D-pad simulation
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col2.button("⬆️", on_click=navigate, args=("pan_up",), use_container_width=True)
        p_col1.button("⬅️", on_click=navigate, args=("pan_left",), use_container_width=True)
        p_col3.button("➡️", on_click=navigate, args=("pan_right",), use_container_width=True)
        p_col2.button("⬇️", on_click=navigate, args=("pan_down",), use_container_width=True)

    with nav_col3:
        st.markdown("**History & Coordinates**")
        h_col1, h_col2, h_col3 = st.columns(3)
        h_col1.button("↩️ Previous", on_click=navigate, args=("undo",), use_container_width=True)
        h_col2.button("🔄 Reset", on_click=navigate, args=("reset",), use_container_width=True)
        h_col3.button("↪️ Next", on_click=navigate, args=("redo",), use_container_width=True)
        
        with st.expander("Go to Custom Coordinates"):
            c_re = st.number_input("Real Center (Re)", value=float(cx), format="%.6f")
            c_im = st.number_input("Imaginary Center (Im)", value=float(cy), format="%.6f")
            c_span = st.number_input("Horizontal Span", value=float(vp.width), format="%.6f")
            st.button("Go", on_click=navigate, args=("custom",), kwargs={"cx": c_re, "cy": c_im, "span": c_span})

    # 3. Viewport Metadata Display
    st.caption(f"**Current Center:** {cx:.6f} + {cy:.6f}i &nbsp;|&nbsp; **Zoom Level:** {zoom_level:,.2f}x &nbsp;|&nbsp; **Boundaries:** Re[{vp.re_min:.5f}, {vp.re_max:.5f}], Im[{vp.im_min:.5f}, {vp.im_max:.5f}]")

    # 4. Computation and Rendering
    try:
        t0 = time.perf_counter()
        escape_times, escaped, unresolved, _ = cached_compute(
            vp.re_min, vp.re_max, vp.im_min, vp.im_max, width, height, max_iter, 2.0
        )
        
        fig = create_mandelbrot_figure(escape_times, unresolved, vp.re_min, vp.re_max, vp.im_min, vp.im_max, cmap)
        png_buf = figure_to_png_buffer(fig)
        t_total = time.perf_counter() - t0
        
        st.success(f"Rendered view in {t_total:.3f} seconds.")
        st.pyplot(fig)
        
        st.download_button(
            "📥 Download Current View (PNG)", 
            data=png_buf, 
            file_name=f"mandelbrot_{zoom_level:.1f}x.png", 
            mime="image/png"
        )
        plt.close(fig)
    except Exception as e:
        st.error(f"Computation failed: {e}")

if __name__ == "__main__":
    main()

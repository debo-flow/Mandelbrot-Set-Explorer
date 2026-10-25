"""
Streamlit Application UI for the Mandelbrot Set Explorer v6.0.0.
Integrates Orbit Trap Visualization, Deep Zoom Precision, and Statistical Diagnostics.
"""
import time
import streamlit as st
import matplotlib.pyplot as plt

from mandelbrot import compute_mandelbrot, MAX_SAFE_PIXELS
from visualization import colorize_fractal, downsample_image, create_mandelbrot_figure, figure_to_png_buffer
from navigation import Viewport, HistoryManager
from precision import evaluate_precision, PrecisionState
from orbit_analysis import compute_single_orbit, plot_single_orbit, calculate_escape_statistics, orbit_to_csv

st.set_page_config(page_title="Mandelbrot Explorer v6.0", page_icon="🌌", layout="wide")

DEFAULT_VIEW = Viewport(-2.5, 1.0, -1.5, 1.5)

if 'history' not in st.session_state:
    st.session_state.history = HistoryManager()
    st.session_state.history.push(DEFAULT_VIEW)
if 'pan_step' not in st.session_state:
    st.session_state.pan_step = 0.20

# Tuples dict hashes effortlessly for Streamlit cache
@st.cache_data(show_spinner=False)
def cached_compute(re_min, re_max, im_min, im_max, w, h, iters, radius, trap_type, trap_params):
    # Reconvert tupled parameters back to dict for the computation engine
    t_params = dict(trap_params) if trap_params else {}
    return compute_mandelbrot(re_min, re_max, im_min, im_max, w, h, iters, radius, trap_type, t_params)

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
    st.title("Orbit-Trap Dynamics & Advanced Analysis Laboratory")
    
    # 1. Output & Rendering Quality
    st.sidebar.header("1. Pipeline Configuration")
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

    # 2. Geometric Orbit Traps
    st.sidebar.header("2. Orbit Trap Geometry")
    trap_type = st.sidebar.selectbox("Trap Type", ["None", "Point", "Circle", "Line", "Cross"])
    trap_params = {}
    if trap_type == "Point":
        trap_params['p'] = complex(st.sidebar.number_input("Point Re", 0.0), st.sidebar.number_input("Point Im", 0.0))
    elif trap_type == "Circle":
        trap_params['p'] = complex(st.sidebar.number_input("Center Re", 0.0), st.sidebar.number_input("Center Im", 0.0))
        trap_params['r'] = st.sidebar.number_input("Radius", 1.0, min_value=0.001)
    elif trap_type == "Line":
        trap_params['A'] = st.sidebar.number_input("Coefficient A", 1.0)
        trap_params['B'] = st.sidebar.number_input("Coefficient B", 0.0)
        trap_params['C'] = st.sidebar.number_input("Constant C", 0.0)
    elif trap_type == "Cross":
        trap_params['x'] = st.sidebar.number_input("Intersect X", 0.0)
        trap_params['y'] = st.sidebar.number_input("Intersect Y", 0.0)

    # 3. Visualization Settings
    st.sidebar.header("3. Visualization Styling")
    mode_list = ["Smooth", "Classic", "Orbit Trap", "Orbit Trap Blend", "Logarithmic", "Binary", "Grayscale"]
    mode = st.sidebar.selectbox("Visualization Mode", mode_list)
    cmap = st.sidebar.selectbox("Color Map", ["magma", "inferno", "plasma", "viridis", "turbo", "twilight", "hot", "cool"])
    interior_color = st.sidebar.selectbox("Interior Color", ["black", "white", "darkblue"])

    raw_vp = st.session_state.history.current
    vp = raw_vp.enforce_aspect_ratio(out_w, out_h)
    cx, cy = vp.center
    
    internal_w = out_w * ss_factor
    internal_h = out_h * ss_factor
    internal_pixels = internal_w * internal_h
    zoom_level = DEFAULT_VIEW.width / vp.width
    
    if internal_pixels > MAX_SAFE_PIXELS:
        st.error(f"Memory limit exceeded: {internal_pixels:,} pixels. Max allowed: {MAX_SAFE_PIXELS:,}.")
        st.stop()

    tab_fractal, tab_stats, tab_orbit = st.tabs(["🌌 Fractal Viewer", "📊 Escape-Time Analysis", "🔍 Single Orbit Inspector"])

    with tab_fractal:
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
        
        diagnostics = evaluate_precision(vp.re_min, vp.re_max, vp.im_min, vp.im_max, internal_w, internal_h)
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Relative Zoom", f"{zoom_level:,.1e}x")
        metrics_cols[1].metric("Real Span", f"{vp.width:.2e}")
        metrics_cols[2].metric("Pixel Spacing", f"{diagnostics.horizontal_step:.2e}")
        metrics_cols[3].metric("Float64 Limit", f"{diagnostics.re_spacing:.2e}")
        
        if diagnostics.state != PrecisionState.NORMAL:
            st.warning(f"⚠️ **{diagnostics.state.value}**: {diagnostics.message}")

        # Rendering Pipeline
        try:
            t0 = time.perf_counter()
            escape_times, smooth_times, trap_distances, escaped, unresolved, _ = cached_compute(
                vp.re_min, vp.re_max, vp.im_min, vp.im_max, internal_w, internal_h, max_iter, 2.0,
                trap_type, tuple(trap_params.items())
            )
            t_comp = time.perf_counter()
            
            raw_img, sm = colorize_fractal(escape_times, smooth_times, trap_distances, unresolved, mode, cmap, interior_color)
            t_color = time.perf_counter()
            
            final_img = downsample_image(raw_img, out_w, out_h, resample_filter)
            t_sample = time.perf_counter()
            
            fig = create_mandelbrot_figure(final_img, sm, vp.re_min, vp.re_max, vp.im_min, vp.im_max, mode)
            
            metadata = {
                "Project": "Mandelbrot-Set-Explorer v6.0.0",
                "Mode": mode,
                "Supersampling": f"{ss_factor}x",
                "Re_Center": str(cx),
                "Im_Center": str(cy),
                "Trap_Type": trap_type,
                "PrecisionState": diagnostics.state.value
            }
            
            png_buf = figure_to_png_buffer(fig, metadata)
            t_total = time.perf_counter() - t0
            
            st.caption(f"⏱️ **Performance:** Math {t_comp-t0:.2f}s | Map {t_color-t_comp:.2f}s | Filter {t_sample-t_color:.2f}s | Total {t_total:.2f}s")
            st.pyplot(fig)
            
            st.download_button("📥 Download High-Res Render", data=png_buf, file_name=f"mandelbrot_{zoom_level:.1e}x.png", mime="image/png")
            plt.close(fig)
        except Exception as e:
            st.error(f"Execution failed: {e}")

    with tab_stats:
        st.subheader("Escape-Time & Render Statistics")
        if 'escape_times' in locals():
            stats = calculate_escape_statistics(escape_times, escaped, unresolved)
            
            scol1, scol2, scol3 = st.columns(3)
            scol1.metric("Total Pixels Evaluated", f"{stats['total_pixels']:,}")
            scol2.metric("Escaped Pixels", f"{stats['escaped_count']:,} ({stats['pct_escaped']:.1f}%)")
            scol3.metric("Unresolved Pixels", f"{stats['unresolved_count']:,} ({stats['pct_unresolved']:.1f}%)")
            
            scol4, scol5, scol6 = st.columns(3)
            scol4.metric("Min Escape Iteration", stats['min_iter'])
            scol5.metric("Max Escape Iteration", stats['max_iter'])
            scol6.metric("Mean Escape Iteration", f"{stats['mean_iter']:.2f}")
            
            st.info("""
            **Mathematical Note:** 
            The percentage of 'unresolved' pixels represents points that did not mathematically escape 
            within the finite iteration limit currently selected in the sidebar. This is a finite numerical 
            approximation, and does not guarantee absolute, mathematically rigorous membership in the Mandelbrot Set.
            """)
        else:
            st.warning("Please generate a fractal in the viewer to analyze its statistics.")

    with tab_orbit:
        st.subheader("Single-Orbit Inspector")
        st.write("Track the exact orbital path $z_{n+1} = z_n^2 + c$ for a specific complex coordinate.")
        
        ocol1, ocol2, ocol3 = st.columns(3)
        with ocol1:
            orb_re = st.number_input("Target Real (Re)", value=0.0, format="%.6f")
        with ocol2:
            orb_im = st.number_input("Target Imaginary (Im)", value=0.0, format="%.6f")
        with ocol3:
            orb_iter = st.number_input("Analysis Iterations", value=50, max_value=1000)
            
        if st.button("🚀 Calculate & Plot Orbit Path"):
            target_c = complex(orb_re, orb_im)
            orbit_data = compute_single_orbit(target_c, orb_iter, 2.0)
            
            fig_orb = plot_single_orbit(orbit_data, target_c)
            st.pyplot(fig_orb)
            plt.close(fig_orb)
            
            st.write("### Orbit Data Table")
            st.dataframe(orbit_data, use_container_width=True)
            
            csv_str = orbit_to_csv(orbit_data)
            st.download_button(
                "📥 Export Orbit to CSV", 
                data=csv_str, 
                file_name=f"orbit_Re{orb_re}_Im{orb_im}.csv", 
                mime="text/csv"
            )

if __name__ == "__main__":
    main()

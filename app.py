"""
Streamlit Application UI for the Mandelbrot Set Explorer v7.0.0.
Integrates Julia Set exploration, Dual-Plane Connection mode, and Advanced Statistics.
"""
import time
import streamlit as st
import matplotlib.pyplot as plt

from mandelbrot import compute_mandelbrot, MAX_SAFE_PIXELS
from julia import compute_julia
from visualization import colorize_fractal, downsample_image, create_fractal_figure, figure_to_png_buffer
from navigation import Viewport, HistoryManager
from precision import evaluate_precision, PrecisionState
from orbit_analysis import compute_single_orbit, plot_single_orbit, calculate_escape_statistics, orbit_to_csv

st.set_page_config(page_title="Fractal Explorer v7.0", page_icon="🌌", layout="wide")

DEFAULT_VIEW = Viewport(-2.5, 1.0, -1.5, 1.5)
JULIA_DEFAULT = Viewport(-2.0, 2.0, -2.0, 2.0)

if 'history' not in st.session_state:
    st.session_state.history = HistoryManager()
    st.session_state.history.push(DEFAULT_VIEW)
if 'pan_step' not in st.session_state:
    st.session_state.pan_step = 0.20

# Cached Math Engines
@st.cache_data(show_spinner=False)
def cached_mandelbrot(re_min, re_max, im_min, im_max, w, h, iters, radius, trap_type, trap_params):
    t_params = dict(trap_params) if trap_params else {}
    return compute_mandelbrot(re_min, re_max, im_min, im_max, w, h, iters, radius, trap_type, t_params)

@st.cache_data(show_spinner=False)
def cached_julia(re_min, re_max, im_min, im_max, w, h, iters, radius, trap_type, trap_params, c):
    t_params = dict(trap_params) if trap_params else {}
    return compute_julia(re_min, re_max, im_min, im_max, w, h, iters, radius, trap_type, t_params, c)

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

def render_fractal_pipeline(engine_func, vp, out_w, out_h, ss_factor, resample_filter, mode, cmap, interior_color, title, marker_c=None):
    internal_w, internal_h = out_w * ss_factor, out_h * ss_factor
    t0 = time.perf_counter()
    
    # Engine execution handles tuple args from caller
    arrays = engine_func(vp.re_min, vp.re_max, vp.im_min, vp.im_max, internal_w, internal_h)
    escape_times, smooth_times, trap_distances, escaped, unresolved, _ = arrays
    
    raw_img, sm = colorize_fractal(escape_times, smooth_times, trap_distances, unresolved, mode, cmap, interior_color)
    final_img = downsample_image(raw_img, out_w, out_h, resample_filter)
    fig = create_fractal_figure(final_img, sm, vp.re_min, vp.re_max, vp.im_min, vp.im_max, mode, title, marker_c)
    
    t_total = time.perf_counter() - t0
    return fig, arrays, t_total

def main():
    st.title("Mandelbrot–Julia Connection Laboratory")
    
    st.sidebar.header("1. Fractal Domain")
    fractal_type = st.sidebar.radio("Active Explorer", ["Mandelbrot Explorer", "Julia Set Explorer", "Dual Connection Mode"])
    
    c_param = 0j
    if fractal_type in ["Julia Set Explorer", "Dual Connection Mode"]:
        st.sidebar.header("2. Julia Parameter (c)")
        preset = st.sidebar.selectbox("Presets", [
            "Custom", "Circle (c=0)", "Basilica (c=-1)", 
            "Rabbit (c=-0.123+0.745j)", "San Marco (c=-0.75)", 
            "Dendrite (c=1i)", "c = 0.285+0.01j"
        ])
        
        presets_map = {
            "Circle (c=0)": 0j, "Basilica (c=-1)": -1+0j, 
            "Rabbit (c=-0.123+0.745j)": -0.123+0.745j,
            "San Marco (c=-0.75)": -0.75+0j, "Dendrite (c=1i)": 1j,
            "c = 0.285+0.01j": 0.285+0.01j
        }
        c_default = presets_map.get(preset, 0j)
        
        c_re = st.sidebar.number_input("Real Part (a)", value=c_default.real, step=0.01, format="%.4f")
        c_im = st.sidebar.number_input("Imag Part (b)", value=c_default.imag, step=0.01, format="%.4f")
        c_param = complex(c_re, c_im)

    st.sidebar.header("3. Rendering Pipeline")
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

    st.sidebar.header("4. Orbit Trap Geometry")
    trap_type = st.sidebar.selectbox("Trap Type", ["None", "Point", "Circle", "Line", "Cross"])
    trap_params = {}
    if trap_type == "Point":
        trap_params['p'] = complex(st.sidebar.number_input("Point Re", 0.0), st.sidebar.number_input("Point Im", 0.0))
    elif trap_type == "Circle":
        trap_params['p'] = complex(st.sidebar.number_input("Center Re", 0.0), st.sidebar.number_input("Center Im", 0.0))
        trap_params['r'] = st.sidebar.number_input("Radius", 1.0, min_value=0.001)
    elif trap_type == "Line":
        trap_params['A'], trap_params['B'], trap_params['C'] = st.sidebar.number_input("A", 1.0), st.sidebar.number_input("B", 0.0), st.sidebar.number_input("C", 0.0)
    elif trap_type == "Cross":
        trap_params['x'], trap_params['y'] = st.sidebar.number_input("Intersect X", 0.0), st.sidebar.number_input("Intersect Y", 0.0)

    st.sidebar.header("5. Visualization Styling")
    mode = st.sidebar.selectbox("Visualization Mode", ["Smooth", "Classic", "Orbit Trap", "Orbit Trap Blend", "Logarithmic", "Binary", "Grayscale"])
    cmap = st.sidebar.selectbox("Color Map", ["magma", "inferno", "plasma", "viridis", "turbo", "twilight", "hot", "cool"], disabled=(mode in ["Binary", "Grayscale"]))
    interior_color = st.sidebar.selectbox("Interior Color", ["black", "white", "darkblue"])

    raw_vp = st.session_state.history.current
    vp = raw_vp.enforce_aspect_ratio(out_w, out_h)
    cx, cy = vp.center
    internal_pixels = (out_w * ss_factor) * (out_h * ss_factor)
    
    if internal_pixels > MAX_SAFE_PIXELS:
        st.error(f"Memory limit exceeded: {internal_pixels:,} pixels. Max allowed: {MAX_SAFE_PIXELS:,}.")
        st.stop()

    tab_fractal, tab_stats, tab_orbit = st.tabs(["🌌 Fractal Viewer", "📊 Escape-Time Analysis", "🔍 Single Orbit Inspector"])

    with tab_fractal:
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
        
        diagnostics = evaluate_precision(vp.re_min, vp.re_max, vp.im_min, vp.im_max, out_w*ss_factor, out_h*ss_factor)
        if diagnostics.state != PrecisionState.NORMAL:
            st.warning(f"⚠️ **{diagnostics.state.value}**: {diagnostics.message}")

        try:
            if fractal_type == "Mandelbrot Explorer":
                engine = lambda rmn, rmx, imn, imx, w, h: cached_mandelbrot(rmn, rmx, imn, imx, w, h, max_iter, 2.0, trap_type, tuple(trap_params.items()))
                fig, arrays, t_total = render_fractal_pipeline(engine, vp, out_w, out_h, ss_factor, resample_filter, mode, cmap, interior_color, "Mandelbrot Set")
                st.caption(f"⏱️ Total Rendering Time: {t_total:.2f}s")
                st.pyplot(fig)
                png_buf = figure_to_png_buffer(fig)
                st.download_button("📥 Download Mandelbrot Render", data=png_buf, file_name="mandelbrot.png", mime="image/png")

            elif fractal_type == "Julia Set Explorer":
                engine = lambda rmn, rmx, imn, imx, w, h: cached_julia(rmn, rmx, imn, imx, w, h, max_iter, 2.0, trap_type, tuple(trap_params.items()), c_param)
                fig, arrays, t_total = render_fractal_pipeline(engine, vp, out_w, out_h, ss_factor, resample_filter, mode, cmap, interior_color, f"Julia Set (c={c_param})")
                st.caption(f"⏱️ Total Rendering Time: {t_total:.2f}s")
                st.pyplot(fig)
                png_buf = figure_to_png_buffer(fig, {"c": str(c_param)})
                st.download_button("📥 Download Julia Render", data=png_buf, file_name=f"julia_{c_param}.png", mime="image/png")

            elif fractal_type == "Dual Connection Mode":
                st.info("💡 **Mathematical Connection:** The Mandelbrot set (left) catalogs all bounded Julia sets. Selecting a parameter `c` inside the Mandelbrot set produces a connected Julia set (right).")
                col_m, col_j = st.columns(2)
                
                with col_m:
                    engine_m = lambda rmn, rmx, imn, imx, w, h: cached_mandelbrot(rmn, rmx, imn, imx, w, h, max_iter, 2.0, trap_type, tuple(trap_params.items()))
                    fig_m, arrays, _ = render_fractal_pipeline(engine_m, vp, out_w//2, out_h//2, ss_factor, resample_filter, mode, cmap, interior_color, "Mandelbrot Space", marker_c=c_param)
                    st.pyplot(fig_m)
                    
                with col_j:
                    j_vp = JULIA_DEFAULT.enforce_aspect_ratio(out_w//2, out_h//2)
                    engine_j = lambda rmn, rmx, imn, imx, w, h: cached_julia(rmn, rmx, imn, imx, w, h, max_iter, 2.0, trap_type, tuple(trap_params.items()), c_param)
                    fig_j, arrays_j, _ = render_fractal_pipeline(engine_j, j_vp, out_w//2, out_h//2, ss_factor, resample_filter, mode, cmap, interior_color, f"Julia (c={c_param})")
                    st.pyplot(fig_j)
                    
        except Exception as e:
            st.error(f"Execution failed: {e}")

    with tab_stats:
        st.subheader("Escape-Time & Render Statistics")
        if 'arrays' in locals():
            escape_times, _, _, escaped, unresolved, _ = arrays
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
        st.write("Track the exact mathematical path of $z_{n+1} = z_n^2 + c$.")
        
        ocol1, ocol2, ocol3, ocol4 = st.columns(4)
        with ocol1:
            orb_re = st.number_input("Parameter (c) Real", value=c_param.real if fractal_type != "Mandelbrot Explorer" else 0.0, format="%.6f")
        with ocol2:
            orb_im = st.number_input("Parameter (c) Imag", value=c_param.imag if fractal_type != "Mandelbrot Explorer" else 0.0, format="%.6f")
        with ocol3:
            z0_re = st.number_input("Initial z0 Real", value=0.0, format="%.6f")
            z0_im = st.number_input("Initial z0 Imag", value=0.0, format="%.6f")
        with ocol4:
            orb_iter = st.number_input("Analysis Iterations", value=50, max_value=1000)
            
        if st.button("🚀 Calculate & Plot Orbit Path"):
            target_c = complex(orb_re, orb_im)
            target_z0 = complex(z0_re, z0_im)
            orbit_data = compute_single_orbit(target_c, target_z0, orb_iter, 2.0)
            
            fig_orb = plot_single_orbit(orbit_data, target_c, target_z0)
            st.pyplot(fig_orb)
            plt.close(fig_orb)
            
            csv_str = orbit_to_csv(orbit_data)
            st.download_button("📥 Export Orbit CSV", data=csv_str, file_name=f"orbit_c{target_c}.csv", mime="text/csv")

if __name__ == "__main__":
    main()

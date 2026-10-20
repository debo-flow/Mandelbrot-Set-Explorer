"""
Streamlit Application UI for the Mandelbrot Set Explorer.
"""
import time
import streamlit as st
import matplotlib.pyplot as plt
from mandelbrot import compute_mandelbrot
from visualization import create_mandelbrot_figure, figure_to_png_buffer

st.set_page_config(
    page_title="Mandelbrot Set Explorer", 
    page_icon="🌌", 
    layout="wide"
)

# Wrapper to leverage Streamlit's caching for expensive mathematical ops
@st.cache_data(show_spinner=False)
def cached_compute(re_min, re_max, im_min, im_max, w, h, iters, radius):
    return compute_mandelbrot(re_min, re_max, im_min, im_max, w, h, iters, radius)

def main():
    st.title("Foundational Mandelbrot Set Visualization Laboratory")
    st.markdown("""
    Explore the complex dynamics of the mathematical iteration: 
    $$ z_{n+1} = z_n^2 + c $$
    where $z_0 = 0$. Points colored black are **unresolved** (they have not escaped the boundary within the iteration limit), while colored points have **escaped** to infinity.
    """)

    st.sidebar.header("Coordinate Boundaries")
    
    # Configurable parameters with safe bounds
    re_min = st.sidebar.number_input("Real Min", value=-2.5, step=0.1)
    re_max = st.sidebar.number_input("Real Max", value=1.0, step=0.1)
    im_min = st.sidebar.number_input("Imaginary Min", value=-1.5, step=0.1)
    im_max = st.sidebar.number_input("Imaginary Max", value=1.5, step=0.1)
    
    st.sidebar.header("Resolution & Computation")
    width = st.sidebar.number_input("Width (pixels)", value=800, min_value=100, max_value=2000, step=100)
    height = st.sidebar.number_input("Height (pixels)", value=800, min_value=100, max_value=2000, step=100)
    max_iter = st.sidebar.slider("Maximum Iterations", min_value=10, max_value=1000, value=100, step=10)
    cmap = st.sidebar.selectbox("Color Map", ["magma", "inferno", "plasma", "viridis", "cividis", "twilight"])

    if st.sidebar.button("Generate Visualization", type="primary"):
        try:
            start_time = time.perf_counter()
            
            with st.spinner("Computing complex orbits..."):
                escape_times, escaped, unresolved, _ = cached_compute(
                    re_min, re_max, im_min, im_max, width, height, max_iter, 2.0
                )
            
            with st.spinner("Rendering scientific visualization..."):
                fig = create_mandelbrot_figure(escape_times, unresolved, re_min, re_max, im_min, im_max, cmap)
                png_buffer = figure_to_png_buffer(fig)
            
            end_time = time.perf_counter()
            
            st.success(f"Computation & Rendering completed in {end_time - start_time:.3f} seconds.")
            
            # Display results summary
            stats_cols = st.columns(3)
            stats_cols[0].metric("Total Points", f"{width * height:,}")
            stats_cols[1].metric("Escaped Points", f"{escaped.sum():,}")
            stats_cols[2].metric("Unresolved Points", f"{unresolved.sum():,}")
            
            # Display Image
            st.pyplot(fig)
            
            # Download Button
            st.download_button(
                label="📥 Download High-Resolution PNG",
                data=png_buffer,
                file_name=f"mandelbrot_{re_min}_{re_max}_{max_iter}iter.png",
                mime="image/png"
            )
            
            # Close figure to prevent memory leak
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Error during computation: {str(e)}")

if __name__ == "__main__":
    main()

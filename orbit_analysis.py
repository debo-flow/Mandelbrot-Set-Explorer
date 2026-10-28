"""
Orbit Analysis and Statistics Module.
Handles Single-Orbit numerical tracking, matplotlib plotting, and whole-grid escape analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def compute_single_orbit(c: complex, z0: complex = 0j, max_iter: int = 50, escape_radius: float = 2.0) -> List[Dict[str, Any]]:
    """Calculates the specific finite orbit of a complex parameter c with initial condition z0."""
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
        
    orbit = []
    z = z0
    
    for i in range(max_iter + 1):
        mag = abs(z)
        escaped = mag > escape_radius
        
        orbit.append({
            "iteration": i,
            "real": z.real,
            "imag": z.imag,
            "magnitude": mag,
            "escaped": escaped
        })
        
        if escaped:
            break
            
        z = z**2 + c
        
    return orbit

def plot_single_orbit(orbit_data: List[Dict[str, Any]], c: complex, z0: complex = 0j) -> plt.Figure:
    """Renders a mathematical 2D plot of the orbit sequence."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    reals = [pt["real"] for pt in orbit_data]
    imags = [pt["imag"] for pt in orbit_data]
    
    ax.plot(reals, imags, marker='o', linestyle='-', color='blue', alpha=0.6, label="Orbit Path")
    
    # Mark start point
    ax.plot(reals[0], imags[0], marker='s', color='green', markersize=8, label="z_0 (Start)")
    
    # Mark termination point appropriately
    if orbit_data[-1]["escaped"]:
        ax.plot(reals[-1], imags[-1], marker='X', color='red', markersize=10, label="Escaped")
    else:
        ax.plot(reals[-1], imags[-1], marker='*', color='purple', markersize=10, label="Unresolved (Bounded)")
        
    ax.set_xlabel("Real Axis")
    ax.set_ylabel("Imaginary Axis")
    ax.set_title(f"Orbit Sequence\nc = {c.real} + {c.imag}j | z_0 = {z0.real} + {z0.imag}j")
    
    # Circle representing the escape boundary
    circle = plt.Circle((0, 0), 2.0, color='red', fill=False, linestyle='--', alpha=0.5, label='Escape Radius (2.0)')
    ax.add_patch(circle)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig

def calculate_escape_statistics(
    escape_times: np.ndarray, 
    escaped_mask: np.ndarray, 
    unresolved_mask: np.ndarray
) -> Dict[str, Any]:
    """Calculates numerical escape-time diagnostics strictly from computed arrays."""
    total_pixels = escape_times.size
    escaped_count = int(np.sum(escaped_mask))
    unresolved_count = int(np.sum(unresolved_mask))
    
    pct_escaped = (escaped_count / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    pct_unresolved = (unresolved_count / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    
    if escaped_count > 0:
        valid_escape_times = escape_times[escaped_mask]
        min_iter = int(np.min(valid_escape_times))
        max_iter = int(np.max(valid_escape_times))
        mean_iter = float(np.mean(valid_escape_times))
    else:
        min_iter = max_iter = 0
        mean_iter = 0.0
        
    return {
        "total_pixels": total_pixels,
        "escaped_count": escaped_count,
        "unresolved_count": unresolved_count,
        "pct_escaped": pct_escaped,
        "pct_unresolved": pct_unresolved,
        "min_iter": min_iter,
        "max_iter": max_iter,
        "mean_iter": mean_iter
    }

def orbit_to_csv(orbit_data: List[Dict[str, Any]]) -> str:
    """Safely converts orbit dict data to a CSV formatted string without Pandas."""
    header = "iteration,real,imaginary,magnitude,escaped\n"
    rows = [f"{pt['iteration']},{pt['real']},{pt['imag']},{pt['magnitude']},{pt['escaped']}" for pt in orbit_data]
    return header + "\n".join(rows)

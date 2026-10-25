import pytest
import numpy as np
import matplotlib.pyplot as plt
from orbit_analysis import compute_single_orbit, calculate_escape_statistics, orbit_to_csv, plot_single_orbit

def test_compute_single_orbit_escaped():
    """Test orbit calculation tracking for a known escaping point c = 1.0"""
    c = 1.0 + 0.0j
    orbit = compute_single_orbit(c, max_iter=10)
    
    # z0=0, z1=1, z2=2, z3=5 (escapes)
    assert len(orbit) == 4
    assert orbit[0]['iteration'] == 0
    assert orbit[0]['magnitude'] == 0.0
    
    # Check escape flag termination
    assert orbit[-1]['escaped'] == True
    assert orbit[-2]['escaped'] == False

def test_compute_single_orbit_bounded():
    """Test orbit calculation for bounded point c = 0.0"""
    orbit = compute_single_orbit(0j, max_iter=5)
    
    assert len(orbit) == 6 # Includes 0 to 5
    assert orbit[-1]['escaped'] == False
    assert all(pt['magnitude'] == 0.0 for pt in orbit)

def test_compute_single_orbit_invalid():
    with pytest.raises(ValueError):
        compute_single_orbit(1j, max_iter=0)

def test_calculate_escape_statistics():
    """Ensures statistical aggregation correctly handles div/0 protection and counts."""
    e_times = np.array([1, 5, 0, 0])
    escaped = np.array([True, True, False, False])
    unresolved = np.array([False, False, True, True])
    
    stats = calculate_escape_statistics(e_times, escaped, unresolved)
    
    assert stats['total_pixels'] == 4
    assert stats['escaped_count'] == 2
    assert stats['unresolved_count'] == 2
    assert stats['pct_escaped'] == 50.0
    assert stats['min_iter'] == 1
    assert stats['max_iter'] == 5
    assert stats['mean_iter'] == 3.0

def test_calculate_escape_statistics_empty():
    """Test empty/zero valid cases to prevent ZeroDivisionError."""
    stats = calculate_escape_statistics(np.array([]), np.array([]), np.array([]))
    assert stats['pct_escaped'] == 0.0
    assert stats['mean_iter'] == 0.0

def test_orbit_to_csv():
    """Ensures raw CSV generation string output is correct."""
    orbit = [{"iteration": 0, "real": 0.0, "imag": 0.0, "magnitude": 0.0, "escaped": False}]
    csv_out = orbit_to_csv(orbit)
    
    assert "iteration,real,imaginary,magnitude,escaped" in csv_out
    assert "0,0.0,0.0,0.0,False" in csv_out

def test_plot_single_orbit():
    """Ensures the orbit plot compiles successfully into a matplotlib figure."""
    orbit = [{"iteration": 0, "real": 0.0, "imag": 0.0, "magnitude": 0.0, "escaped": False}]
    fig = plot_single_orbit(orbit, 0j)
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

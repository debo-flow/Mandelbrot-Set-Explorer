import pytest
import numpy as np
import matplotlib.pyplot as plt
from orbit_analysis import compute_single_orbit, calculate_escape_statistics, orbit_to_csv, plot_single_orbit

def test_compute_single_orbit_escaped():
    """Test orbit calculation tracking for a known escaping point c = 1.0, z0 = 0"""
    c = 1.0 + 0.0j
    orbit = compute_single_orbit(c, z0=0j, max_iter=10)
    
    assert len(orbit) == 4
    assert orbit[0]['iteration'] == 0
    assert orbit[0]['magnitude'] == 0.0
    assert orbit[-1]['escaped'] == True
    assert orbit[-2]['escaped'] == False

def test_compute_single_orbit_julia():
    """Test orbit calculation for a Julia setup where z0 escapes under fixed c."""
    c = 0j
    orbit = compute_single_orbit(c, z0=2j, max_iter=5)
    # z0=2j (mag=2), z1=-4 (mag=4 > 2 -> escapes immediately)
    assert len(orbit) == 2
    assert orbit[-1]['escaped'] == True

def test_compute_single_orbit_bounded():
    orbit = compute_single_orbit(0j, z0=0j, max_iter=5)
    assert len(orbit) == 6
    assert orbit[-1]['escaped'] == False

def test_compute_single_orbit_invalid():
    with pytest.raises(ValueError):
        compute_single_orbit(1j, max_iter=0)

def test_calculate_escape_statistics():
    e_times = np.array([1, 5, 0, 0])
    escaped = np.array([True, True, False, False])
    unresolved = np.array([False, False, True, True])
    
    stats = calculate_escape_statistics(e_times, escaped, unresolved)
    assert stats['pct_escaped'] == 50.0
    assert stats['min_iter'] == 1
    assert stats['max_iter'] == 5

def test_orbit_to_csv():
    orbit = [{"iteration": 0, "real": 0.0, "imag": 0.0, "magnitude": 0.0, "escaped": False}]
    csv_out = orbit_to_csv(orbit)
    assert "0,0.0,0.0,0.0,False" in csv_out

def test_plot_single_orbit():
    orbit = [{"iteration": 0, "real": 0.0, "imag": 0.0, "magnitude": 0.0, "escaped": False}]
    fig = plot_single_orbit(orbit, 0j, 0j)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

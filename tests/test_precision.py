import pytest
import numpy as np
from precision import evaluate_precision, PrecisionState

def test_precision_normal_view():
    """Standard view should be perfectly Normal."""
    diag = evaluate_precision(-2.5, 1.0, -1.5, 1.5, width=800, height=800)
    assert diag.state == PrecisionState.NORMAL
    assert "optimal" in diag.message.lower()

def test_precision_warning_view():
    """Deep zoom reaching the 100x spacing boundary triggers a Warning."""
    cx, cy = -0.75, 0.1
    span = 1e-13
    diag = evaluate_precision(
        cx - span/2, cx + span/2, cy - span/2, cy + span/2, 800, 800
    )
    assert diag.state == PrecisionState.WARNING
    assert "approaching float64 limits" in diag.message

def test_precision_limitation_reached():
    """Extreme zoom where pixels mathematically map to identical float coordinates."""
    cx, cy = -0.75, 0.1
    span = 1e-16 
    diag = evaluate_precision(
        cx - span/2, cx + span/2, cy - span/2, cy + span/2, 800, 800
    )
    assert diag.state == PrecisionState.LIMITATION
    assert "identical floating-point values" in diag.message

def test_precision_zero_center():
    """Zooming precisely at 0.0 has a much smaller spacing constraint."""
    span = 1e-100
    diag = evaluate_precision(-span, span, -span, span, 800, 800)
    assert diag.state == PrecisionState.NORMAL

def test_invalid_dimensions():
    with pytest.raises(ValueError):
        evaluate_precision(-1, 1, -1, 1, 1, 800)

def test_nan_inputs():
    with pytest.raises(ValueError):
        evaluate_precision(float('nan'), 1, -1, 1, 800, 800)

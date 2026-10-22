import pytest
from navigation import Viewport, HistoryManager

def test_viewport_zoom():
    vp = Viewport(-2.0, 2.0, -2.0, 2.0)
    vp_in = vp.zoom(2.0)
    assert vp_in.width == 2.0
    assert vp_in.height == 2.0
    assert vp_in.center == (0.0, 0.0)

    vp_out = vp.zoom(0.5)
    assert vp_out.width == 8.0
    assert vp_out.height == 8.0
    assert vp_out.center == (0.0, 0.0)

def test_extreme_decimal_zoom_preservation():
    """Ensures Viewport correctly preserves high-precision decimal centers during zoom operations."""
    cx, cy = -0.743643887037151, 0.131825904205330
    span = 1e-12
    vp = Viewport(cx - span, cx + span, cy - span, cy + span)
    
    # Zoom deeply
    vp_deep = vp.zoom(100.0)
    new_cx, new_cy = vp_deep.center
    
    # The center must remain exactly identical down to float64 limits
    import math
    assert math.isclose(new_cx, cx, rel_tol=1e-15)
    assert math.isclose(new_cy, cy, rel_tol=1e-15)

def test_invalid_zoom():
    vp = Viewport(0, 1, 0, 1)
    with pytest.raises(ValueError):
        vp.zoom(-1.0)
    with pytest.raises(ValueError):
        vp.zoom(0.0)

def test_viewport_pan():
    vp = Viewport(0.0, 10.0, 0.0, 10.0)
    vp_right = vp.pan(dx_frac=0.2, dy_frac=0.0)
    assert vp_right.re_min == 2.0
    assert vp_right.re_max == 12.0
    assert vp_right.center == (7.0, 5.0)

def test_enforce_aspect_ratio():
    vp = Viewport(-2.0, 2.0, -2.0, 2.0) 
    vp_adjusted = vp.enforce_aspect_ratio(800, 400)
    assert vp_adjusted.width == 4.0
    assert vp_adjusted.height == 2.0
    assert vp_adjusted.center == (0.0, 0.0)

def test_history_manager():
    hm = HistoryManager(limit=3)
    vp1 = Viewport(0, 1, 0, 1)
    vp2 = Viewport(0, 2, 0, 2)
    vp3 = Viewport(0, 3, 0, 3)
    vp4 = Viewport(0, 4, 0, 4)

    hm.push(vp1)
    hm.push(vp2)
    assert hm.current == vp2

    hm.undo()
    assert hm.current == vp1
    hm.redo()
    assert hm.current == vp2

    hm.push(vp3)
    hm.push(vp4) 
    assert len(hm.history) == 3
    assert hm.current == vp4

    hm.undo()
    hm.undo()
    assert hm.current == vp2
    assert not hm.undo()

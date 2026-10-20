import pytest
from navigation import Viewport, HistoryManager

def test_viewport_zoom():
    vp = Viewport(-2.0, 2.0, -2.0, 2.0)
    # Zoom In
    vp_in = vp.zoom(2.0)
    assert vp_in.width == 2.0
    assert vp_in.height == 2.0
    assert vp_in.center == (0.0, 0.0) # Center preserved

    # Zoom Out
    vp_out = vp.zoom(0.5)
    assert vp_out.width == 8.0
    assert vp_out.height == 8.0
    assert vp_out.center == (0.0, 0.0)

def test_invalid_zoom():
    vp = Viewport(0, 1, 0, 1)
    with pytest.raises(ValueError):
        vp.zoom(-1.0)
    with pytest.raises(ValueError):
        vp.zoom(0.0)

def test_viewport_pan():
    vp = Viewport(0.0, 10.0, 0.0, 10.0)
    # Pan right by 20%
    vp_right = vp.pan(dx_frac=0.2, dy_frac=0.0)
    assert vp_right.re_min == 2.0
    assert vp_right.re_max == 12.0
    assert vp_right.center == (7.0, 5.0)

    # Pan up by 10%
    vp_up = vp.pan(dx_frac=0.0, dy_frac=0.1)
    assert vp_up.im_min == 1.0
    assert vp_up.im_max == 11.0

def test_enforce_aspect_ratio():
    # Real span = 4.0, Im span = 4.0. Ratio is 1:1
    vp = Viewport(-2.0, 2.0, -2.0, 2.0) 
    
    # Force output to 800w x 400h (2:1 aspect ratio -> width is twice height)
    # So Target Ratio (h/w) = 0.5
    vp_adjusted = vp.enforce_aspect_ratio(800, 400)
    
    assert vp_adjusted.width == 4.0   # Real width remains anchor
    assert vp_adjusted.height == 2.0  # Height is scaled down
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

    # Test limit enforcement (limit is 3)
    hm.push(vp3)
    hm.push(vp4) 
    assert len(hm.history) == 3
    assert hm.current == vp4

    # The oldest (vp1) should be gone
    hm.undo()
    hm.undo()
    assert hm.current == vp2
    assert not hm.undo() # Cannot undo past limit

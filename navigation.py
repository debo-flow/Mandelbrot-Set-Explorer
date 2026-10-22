"""
Complex Plane Navigation Module.
Handles Viewport mathematics, aspect ratio enforcement, and strict float precision validation.
"""
import math
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Viewport:
    re_min: float
    re_max: float
    im_min: float
    im_max: float

    def __post_init__(self):
        self._validate()

    def _validate(self):
        for v in (self.re_min, self.re_max, self.im_min, self.im_max):
            if math.isnan(v) or math.isinf(v):
                raise ValueError("Viewport boundaries must be finite numbers.")
        if self.re_max <= self.re_min:
            raise ValueError("re_max must be strictly greater than re_min.")
        if self.im_max <= self.im_min:
            raise ValueError("im_max must be strictly greater than im_min.")

    @property
    def center(self) -> Tuple[float, float]:
        return (self.re_min + self.re_max) / 2.0, (self.im_min + self.im_max) / 2.0

    @property
    def width(self) -> float:
        return self.re_max - self.re_min

    @property
    def height(self) -> float:
        return self.im_max - self.im_min

    def enforce_aspect_ratio(self, px_width: int, px_height: int) -> 'Viewport':
        """
        Adjusts the vertical boundaries to match the exact pixel aspect ratio,
        preventing image stretching while preserving the intended center coordinate.
        """
        if px_width <= 0 or px_height <= 0:
            raise ValueError("Pixel dimensions must be positive.")
            
        target_ratio = px_height / px_width
        cx, cy = self.center
        
        new_width = self.width
        new_height = new_width * target_ratio
        
        return Viewport(
            re_min=cx - new_width / 2.0,
            re_max=cx + new_width / 2.0,
            im_min=cy - new_height / 2.0,
            im_max=cy + new_height / 2.0
        )

    def zoom(self, factor: float) -> 'Viewport':
        if factor <= 0 or math.isnan(factor) or math.isinf(factor):
            raise ValueError("Zoom factor must be a finite positive number.")
            
        cx, cy = self.center
        new_w = self.width / factor
        new_h = self.height / factor
        
        return Viewport(
            re_min=cx - new_w / 2.0,
            re_max=cx + new_w / 2.0,
            im_min=cy - new_h / 2.0,
            im_max=cy + new_h / 2.0
        )

    def pan(self, dx_frac: float, dy_frac: float) -> 'Viewport':
        shift_x = self.width * dx_frac
        shift_y = self.height * dy_frac
        
        return Viewport(
            re_min=self.re_min + shift_x,
            re_max=self.re_max + shift_x,
            im_min=self.im_min + shift_y,
            im_max=self.im_max + shift_y
        )

class HistoryManager:
    """Manages the navigation history stack to allow undo/redo operations."""
    def __init__(self, limit: int = 50):
        self.history: List[Viewport] = []
        self.current_index: int = -1
        self.limit = limit

    def push(self, viewport: Viewport):
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
            
        self.history.append(viewport)
        if len(self.history) > self.limit:
            self.history.pop(0)
        else:
            self.current_index += 1

    def undo(self) -> bool:
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def redo(self) -> bool:
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            return True
        return False

    @property
    def current(self) -> Viewport:
        if self.current_index >= 0:
            return self.history[self.current_index]
        raise IndexError("History is empty.")

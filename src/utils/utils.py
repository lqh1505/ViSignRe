"""Shared helpers and display color constants."""

import time
import numpy as np
from collections import deque
from config import Config

C_GREEN = (0, 220, 80)
C_RED = (50, 50, 230)
C_YELLOW = (0, 220, 220)
C_GRAY = (140, 140, 140)
C_WHITE = (255, 255, 255)
C_CYAN = (255, 255, 0)


class FPSCounter:
    def __init__(self, smoothing: int = 30):
        self._times = deque(maxlen=smoothing)
        self._last = time.time()

    def tick(self) -> float:
        now = time.time()
        self._times.append(now - self._last)
        self._last = now
        return 1.0 / (np.mean(self._times) + Config.FPS_EPSILON)

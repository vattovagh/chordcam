"""Drawing on frames."""
import time

import cv2


class FpsCounter:
    """Frames per second, smoothed with an exponential moving average."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.fps = 0.0
        self._last: float | None = None

    def tick(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        if self._last is not None and now > self._last:
            inst = 1.0 / (now - self._last)
            self.fps = inst if self.fps == 0.0 else self.alpha * inst + (1 - self.alpha) * self.fps
        self._last = now
        return self.fps


def draw_fps(frame, fps: float):
    text = f"{fps:5.1f} FPS"
    cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

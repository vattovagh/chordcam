import numpy as np

from chordcam.config import load_config
from chordcam.overlay import FpsCounter, draw_fps


def test_config_defaults():
    cfg = load_config()
    assert cfg["classes"] == ["C", "D", "Em", "F", "G", "none"]
    assert cfg["smoothing"]["ema_alpha"] == 0.3
    assert cfg["paths"]["data"].endswith("/data")  # resolved to an absolute path


def test_fps_counter_converges():
    c = FpsCounter()
    for i in range(100):
        c.tick(now=i / 30)
    assert abs(c.fps - 30) < 0.5


def test_draw_fps_modifies_frame():
    frame = np.zeros((100, 300, 3), np.uint8)
    assert draw_fps(frame, 29.9).any()

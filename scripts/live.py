#!/usr/bin/env python
"""Real-time viewer (Milestone 1: mirrored feed with FPS; chord recognition comes later).

Examples:
  python scripts/live.py                      # built-in webcam
  python scripts/live.py --source clip.mp4
  python scripts/live.py --source "https://www.youtube.com/watch?v=..."
"""
import argparse
import sys

import cv2

from chordcam.capture import VideoSource
from chordcam.config import load_config
from chordcam.overlay import FpsCounter, draw_fps


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0",
                    help="webcam index, video file, or YouTube URL (default: 0)")
    ap.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--width", type=int, help="capture width (default: from config)")
    ap.add_argument("--height", type=int, help="capture height (default: from config)")
    ap.add_argument("--no-mirror", action="store_true", help="don't mirror the displayed image")
    ap.add_argument("--no-display", action="store_true",
                    help="don't open a window (for headless testing)")
    ap.add_argument("--max-frames", type=int, help="stop after this many frames")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    width = args.width or cfg["capture"]["width"]
    height = args.height or cfg["capture"]["height"]
    fps = FpsCounter()
    n = 0
    try:
        with VideoSource(args.source, width, height) as src:
            while True:
                frame = src.read()
                if frame is None:
                    break
                n += 1
                # Mirror for display only; frames sent to a model must stay unflipped.
                view = frame if args.no_mirror else cv2.flip(frame, 1)
                draw_fps(view, fps.tick())
                if not args.no_display:
                    cv2.imshow("chordcam", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if args.max_frames and n >= args.max_frames:
                    break
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    if n == 0:
        print("error: no frames received. For a webcam, allow camera access for your "
              "terminal in System Settings > Privacy & Security > Camera, and check "
              "no other app is using it.", file=sys.stderr)
        return 1
    print(f"{n} frames, final {fps.fps:.1f} FPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

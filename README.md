# chordcam

Webcam guitar chord recognizer. See [SPEC.md](SPEC.md) for the full design and milestones.

## Setup

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .          # makes `chordcam` importable by the scripts
```

Run tests: `.venv/bin/python -m pytest`

## Milestone 1: webcam viewer

```
.venv/bin/python scripts/live.py                          # built-in webcam
.venv/bin/python scripts/live.py --source clip.mp4        # video file
.venv/bin/python scripts/live.py --source "<youtube url>" # via yt-dlp
.venv/bin/python scripts/live.py --help
```

Shows the mirrored feed with an FPS counter; press `q` to quit. The mirroring is display-only.
On macOS, the first run asks for camera permission for your terminal app; if you declined,
enable it in System Settings > Privacy & Security > Camera.

Settings live in `configs/default.yaml`.

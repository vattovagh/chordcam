"""Frame sources: a webcam index, a video file, or a YouTube URL."""
import cv2


def resolve_stream_url(url: str) -> str:
    """Ask yt-dlp for a direct stream URL OpenCV can open."""
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "format": "best[height<=720][ext=mp4]/best[height<=720]/best"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)["url"]


class VideoSource:
    def __init__(self, source: str, width: int | None = None, height: int | None = None):
        self.is_live = source.isdigit()
        if self.is_live:
            self.cap = cv2.VideoCapture(int(source))
            if width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        elif source.startswith(("http://", "https://")):
            self.cap = cv2.VideoCapture(resolve_stream_url(source))
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            hint = (" On macOS, allow camera access for your terminal in "
                    "System Settings > Privacy & Security > Camera." if self.is_live else "")
            raise RuntimeError(f"could not open source {source!r}.{hint}")

    def read(self):
        """Return the next BGR frame, or None at end of stream."""
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

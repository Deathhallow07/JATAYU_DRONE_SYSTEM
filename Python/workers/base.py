"""
Shared scaffolding for the model panes.

A worker is a loop: pull a frame, run a model on it, publish an annotated JPEG
plus whatever structured result the pane needs. Everything model-specific lives
in the subclass; everything else (source handling, throttling, encoding,
reconnection, command dispatch) lives here, so swapping a stub for a real
checkpoint is a change to two methods.

Frames go over the socket as base64 JPEG rather than a separate MJPEG port.
That is slower than raw MJPEG, but it keeps one connection, one auth surface
and one reconnect path, and at the 5-10 fps these panes run at the difference
does not show. If a pane ever needs full frame rate, point it at the pipeline's
own MJPEG server (DISPLAY_MODE="stream", STREAM_PORT 8080) instead.
"""

import base64
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bus import Bus  # noqa: E402

log = logging.getLogger("worker")

JPEG_QUALITY = 80


def encode_jpeg(bgr, quality=JPEG_QUALITY):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


class FrameSource:
    """
    A video file, a camera index, an MJPEG URL, or a still image.

    Stills are held open and re-served so a worker can be demoed against a
    single photograph with no special-casing in the loop.
    """

    IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(self, spec, loop=True, max_width=960):
        self.spec = spec
        self.loop = loop
        self.max_width = max_width
        self._still = None
        self._cap = None

        if isinstance(spec, str) and spec.lower().endswith(self.IMAGE_EXT):
            img = cv2.imread(spec)
            if img is None:
                raise FileNotFoundError(f"cannot read image: {spec}")
            self._still = self._fit(img)
        else:
            src = int(spec) if str(spec).isdigit() else spec
            self._cap = cv2.VideoCapture(src)
            if not self._cap.isOpened():
                raise RuntimeError(f"cannot open video source: {spec}")

    def _fit(self, frame):
        h, w = frame.shape[:2]
        if w <= self.max_width:
            return frame
        scale = self.max_width / w
        return cv2.resize(frame, (self.max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    def read(self):
        if self._still is not None:
            return self._still.copy()

        ok, frame = self._cap.read()
        if not ok:
            if not self.loop:
                return None
            # Rewind works for files; for a camera or a stream it is a no-op
            # and the next read fails again, which the caller treats as EOF.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                return None
        return self._fit(frame)

    def release(self):
        if self._cap is not None:
            self._cap.release()


class BaseWorker:
    """
    Subclasses implement:

        load()              once, before the loop — bring up the model
        infer(frame)        per frame — return (vis_bgr, extra_dict)

    and set NAME to the pane key the frontend listens on.
    """

    NAME = "base"

    def __init__(self, source, url=None, fps=5.0, loop=True, width=960):
        self.bus = Bus(f"worker:{self.NAME}", url) if url else Bus(f"worker:{self.NAME}")
        self.source_spec = source
        self.fps = fps
        self.loop_video = loop
        self.width = width

        self.running = True
        self.paused = False
        self.model = None
        self._stop = threading.Event()
        self._seq = 0

    # -- lifecycle ---------------------------------------------------------

    def load(self):
        raise NotImplementedError

    def infer(self, frame):
        raise NotImplementedError

    def on_command(self, payload):
        """Optional per-worker commands from the browser."""

    # -- status ------------------------------------------------------------

    def status(self, state, **extra):
        self.bus.emit("worker_status", {"worker": self.NAME, "state": state, **extra})

    # -- main loop ---------------------------------------------------------

    def start(self):
        self.bus.connect()

        def _cmd(payload=None):
            payload = payload or {}
            if payload.get("worker") not in (None, self.NAME):
                return
            action = payload.get("action")
            if action == "pause":
                self.paused = True
            elif action == "play":
                self.paused = False
            self.on_command(payload)

        self.bus.on("run_worker", _cmd)
        self.bus.on("plan_path", lambda p=None: self.on_command({"action": "plan_path", **(p or {})}))

        self.status("loading")
        try:
            self.load()
        except Exception as exc:
            log.exception("model load failed")
            self.status("error", error=str(exc))
            return

        try:
            source = FrameSource(self.source_spec, self.loop_video, self.width)
        except Exception as exc:
            log.exception("source open failed")
            self.status("error", error=str(exc))
            return

        self.status("running", source=str(self.source_spec))
        interval = 1.0 / max(0.1, self.fps)

        try:
            while self.running and not self._stop.is_set():
                t0 = time.time()

                if self.paused:
                    time.sleep(0.1)
                    continue

                frame = source.read()
                if frame is None:
                    log.info("source exhausted")
                    break

                try:
                    vis, extra = self.infer(frame)
                except Exception as exc:
                    log.exception("inference failed")
                    self.status("error", error=str(exc))
                    time.sleep(1.0)
                    continue

                jpeg = encode_jpeg(vis)
                if jpeg is None:
                    continue

                self._seq += 1
                self.bus.emit("frame", {
                    "worker": self.NAME,
                    "seq": self._seq,
                    "ts": time.time(),
                    "jpeg": jpeg,
                    "width": vis.shape[1],
                    "height": vis.shape[0],
                    "latencyMs": round((time.time() - t0) * 1000, 1),
                    **(extra or {}),
                })

                time.sleep(max(0.0, interval - (time.time() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            source.release()
            self.status("stopped")
            self.bus.close()

    def stop(self):
        self.running = False
        self._stop.set()


# ---------------------------------------------------------------------------
# Colour helpers shared by the panes
# ---------------------------------------------------------------------------

def normalize01(arr):
    """Min-max to [0,1], flat input included (a constant frame is not an error)."""
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def label_frame(bgr, text, color=(255, 255, 255)):
    """Bottom-left caption with a dark plate behind it, legible over any frame."""
    h = bgr.shape[0]
    cv2.rectangle(bgr, (0, h - 26), (bgr.shape[1], h), (0, 0, 0), -1)
    cv2.putText(bgr, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return bgr

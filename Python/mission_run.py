#!/usr/bin/env python3
"""
One video + one telemetry CSV in, the whole console live.

    python mission_run.py --video clip.mp4 --telem fcb.csv

This is the live counterpart to ingest_video.py. That script processes a video
to completion and *then* writes a run folder for the watcher to replay; this one
drives the console as it goes:

    Mission map      the drone flies the CSV's path, dragging its trail, and a
                     casualty pin drops the instant a GID is created
    Casualties       GID cards appear live, and update on re-identification
    Geolocalisation  the annotated feed — boxes, track ids, GID labels, GPS
    Traversability   seg_worker, on the same frames
    Depth            depth_worker, on the same frames
    Notifications    "Casualty 3 found", "Casualty 3 seen again"

Everything the console shows is emitted by this one process, so there is no
mission_watcher and no replay: what you see is the frame being processed.

    ┌──────────────────────── mission_run.py ────────────────────────┐
    │  video ──┬─► detect + track + ReID ──► gid  ──────┐            │
    │          │         └─► annotated frame ──► detect │            │
    │          └─► MJPEG hub :8090 ─┐                   ├──► relay   │
    │  csv   ────► telemetry @ video time ──────────────┘            │
    └───────────────────────────────┼────────────────────────────────┘
                                    ▼
                    depth_worker / seg_worker subprocesses
                    (--source http://127.0.0.1:8090/)

The hub is why the three feeds stay in step. The workers could each open the
mp4 themselves, but then three decoders advance independently and the panes
show three different moments of the flight. One decode, published once, keeps
depth, traversability and the detections on the same frame.

TIME IS VIDEO TIME, NOT WALL TIME. Telemetry is emitted for the timestamp of
the frame being processed, so on a CPU box running at a fifth of real speed the
drone still sits exactly where it was when that frame was captured. `--realtime`
switches to wall-clock pacing (dropping frames to keep up) for a machine fast
enough to hold it.

Environment: needs `ultralytics` AND `python-socketio` in ONE interpreter.
Here those live in different conda envs, so:

    ~/miniconda3/envs/work_env/bin/pip install "python-socketio[client]"

Then:

    ./Live.sh clip.mp4 fcb.csv          # relay + console + this, in tmux
    python mission_run.py --video clip.mp4 --telem fcb.csv --no-workers
"""

import argparse
import logging
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bus import Bus                       # noqa: E402
from run_writer import RunWriter          # noqa: E402
from mission_watcher import build_gid_payload  # noqa: E402
from telem_track import TelemetryTrack, haversine_m  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("mission")

# Matches the pipeline's crop geometry so the cards look like the real thing.
CROP_W, CROP_H = 128, 256
BBOX_EXPAND = 1.5

DEFAULT_PROMPTS = [
    "person",
    "person lying on the ground",
    "injured person",
    "casualty",
]

# Geo gating on re-identification, from the pipeline's own thresholds: a match
# closer than MATCH is accepted on appearance alone, one beyond FAR is refused
# however similar it looks. Two people in identical uniforms forty metres apart
# are two casualties, and that is the mistake worth engineering against.
GEO_MATCH_M = 3.0
GEO_FAR_M = 7.0

JPEG_QUALITY = 80


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def expand_crop(frame, box):
    """Expanded, aspect-corrected crop, resized to the pipeline's gallery size."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * BBOX_EXPAND, (y2 - y1) * BBOX_EXPAND

    x1 = max(0, int(cx - bw / 2))
    x2 = min(w, int(cx + bw / 2))
    y1 = max(0, int(cy - bh / 2))
    y2 = min(h, int(cy + bh / 2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return cv2.resize(frame[y1:y2, x1:x2], (CROP_W, CROP_H), interpolation=cv2.INTER_LINEAR)


def sharpness(crop):
    """Laplacian variance — the pipeline's own sharpness measure."""
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def medoid(points):
    """
    The point with the least total distance to the others.

    A medoid rather than a mean because GPS outliers are common and one bad fix
    drags a mean off the casualty; the medoid is an actual observed position and
    ignores the outlier entirely. This is what the pipeline reports as a GID's
    final coordinate.
    """
    if not points:
        return None
    if len(points) <= 2:
        return points[0]
    best, best_cost = points[0], float("inf")
    for p in points:
        cost = sum(haversine_m(p[0], p[1], q[0], q[1]) for q in points)
        if cost < best_cost:
            best, best_cost = p, cost
    return best


# ---------------------------------------------------------------------------
# MJPEG hub
# ---------------------------------------------------------------------------

class FrameHub:
    """
    Serves the frames this process decodes, as MJPEG, for the model workers.

    Same protocol the pipeline's DISPLAY_MODE="stream" uses, so the workers need
    no changes: they already accept an MJPEG URL as --source.
    """

    def __init__(self, port):
        self.port = port
        self._frame = None
        self._lock = threading.Condition()
        self._server = None
        self._seq = 0

    def publish(self, bgr):
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return
        with self._lock:
            self._frame = buf.tobytes()
            self._seq += 1
            self._lock.notify_all()

    def _wait(self, last_seq, timeout=5.0):
        with self._lock:
            if self._seq == last_seq:
                self._lock.wait(timeout)
            return self._frame, self._seq

    def start(self):
        hub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_a):
                pass  # the worker's own logging is enough

            def do_GET(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        frame, seq = hub._wait(seq)
                        if frame is None:
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the worker went away; its reconnect brings it back

        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        log.info("frame hub on http://127.0.0.1:%d/", self.port)

    def stop(self):
        if self._server:
            self._server.shutdown()



# ---------------------------------------------------------------------------
# Geolocalisation
# ---------------------------------------------------------------------------

# Where the pipeline's calibrated transformer lives. Reused rather than
# reimplemented: R_BODY_CAM and the ground offset in that file were fitted
# together by least squares over surveyed people, and a re-derivation here
# would be a second copy free to drift from the one the pipeline flies with.
TRANSFORMER_PATHS = (
    os.path.expanduser("~/DK3ROX/Programs/Python/ReID-Pipeline/YOLOE"),
    os.path.expanduser("~/DK3ROX/Programs/Python/ReID-Pipeline/FINAL/deployment_gate_1/v3"),
    os.path.expanduser("~/DK3ROX/Programs/Python/ReID-Pipeline/GPS-Integration-v2"),
)

# Sony FCB, the camera the recordings in this repo were shot on — the same
# matrix Pipeline_yoloe_logs_robust.py uses.
CAMERA_INTRINSICS = np.array([
    [1584.02798, 0.0,        953.27546],
    [0.0,        1581.52473, 518.97355],
    [0.0,        0.0,        1.0],
])


class Geolocaliser:
    """
    Turns a detection box into the casualty's ground position.

    Without this, a casualty's "position" is wherever the DRONE was when it saw
    them. That is wrong in two ways that both matter here: the pin lands on the
    flight path instead of on the person, and — because a moving drone sees the
    same casualty from a different place each time — every sighting looks like a
    different location, so the geo gate can never recognise a return and one
    person becomes a string of GIDs.

    Pixel coordinates must be given in the ORIGINAL frame's resolution. The
    detection loop works on downscaled frames, so it scales them back up first:
    the intrinsics are calibrated to the sensor, not to whatever the console
    happens to be displaying.
    """

    def __init__(self, enabled=True, intrinsics=None, zoom_fallback=1.0):
        self.transformer = None
        self.reason = None
        # Used only for frames whose telemetry row carries no zoom_ratio. A CSV
        # that logs the column drives the intrinsics per frame and never
        # reaches this; 1.0 means "the intrinsics exactly as calibrated".
        self.zoom_fallback = zoom_fallback if zoom_fallback and zoom_fallback > 0 else 1.0
        if not enabled:
            self.reason = "disabled"
            return
        for path in TRANSFORMER_PATHS:
            candidate = os.path.join(path, "coordinate_transformer.py")
            if not os.path.exists(candidate):
                continue
            try:
                sys.path.insert(0, path)
                from coordinate_transformer import CoordinateTransformer
                self.transformer = CoordinateTransformer(
                    intrinsics if intrinsics is not None else CAMERA_INTRINSICS)
                log.info("geolocalisation: using %s", candidate)
                return
            except Exception:
                log.exception("could not load %s", candidate)
        self.reason = "coordinate_transformer.py not found"
        log.warning("geolocalisation unavailable (%s) — casualty positions will "
                    "fall back to the drone's own fix", self.reason)

    @property
    def available(self):
        return self.transformer is not None

    def locate(self, box, state, scale=1.0):
        """
        Ground position for one detection, or None if it cannot be projected.

        `scale` maps the working frame back to the calibrated resolution.
        Returns None whenever the geometry does not close — no attitude in the
        log, the ray pointing at the sky, a non-positive altitude — rather than
        inventing a coordinate, because a confident wrong pin sends a medic to
        the wrong place.
        """
        if not self.available or state is None:
            return None
        roll, pitch, yaw = state.get("roll"), state.get("pitch"), state.get("yaw")
        alt = state.get("alt")
        if None in (roll, pitch, yaw) or alt is None or alt <= 0.5:
            return None

        x1, y1, x2, y2 = (v * scale for v in box)
        # The pipeline's own sample point: horizontally centred, and near the
        # bottom of the box, which is where a person meets the ground.
        u = x1 + 0.5 * (x2 - x1)
        v = y2 - 0.1 * (y2 - y1)

        try:
            result = self.transformer.pixel_to_gps(
                u=u, v=v,
                drone_lat=state["lat"], drone_lon=state["lon"],
                drone_altitude=alt,
                roll=math.radians(roll), pitch=math.radians(pitch),
                yaw=math.radians(yaw),
                ground_altitude=0.0,
                # Per-frame optical zoom: it scales fx/fy, so a clip that
                # zooms mid-flight would otherwise project every frame through
                # the wrong focal length.
                zoom=state.get("zoom") or self.zoom_fallback,
            )
        except Exception:
            log.exception("projection failed")
            return None

        if not result:
            return None
        lat, lon = result[0], result[1]
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        return float(lat), float(lon)


# ---------------------------------------------------------------------------
# Re-identification
# ---------------------------------------------------------------------------

class Embedder:
    """
    Appearance embeddings for re-identification.

    Wraps a TorchScript module (mobileclip2_b.ts ships with this repo) and
    normalises whatever it returns to a unit vector, so matching is a dot
    product regardless of which checkpoint is loaded.

    Without one, `available` is False and GID assignment falls back to geography
    alone — which is stated plainly in mission.log rather than quietly pretended
    over, because it is the difference between recognising a casualty and
    guessing from where they are lying.
    """

    # Tried in order on the first crop, then the winner is kept. A TorchScript
    # module carries no input signature, so the only way to learn what it wants
    # is to offer it something and see whether it throws.
    INPUT_SIZES = ((128, 256), (224, 224), (192, 384))

    def __init__(self, path=None, device="cpu"):
        self.available = False
        self.model = None
        self.device = device
        self.size = None
        if not path:
            return
        if not os.path.exists(path):
            log.warning("ReID checkpoint not found: %s", path)
            return
        try:
            import torch
            self.torch = torch
            self.model = torch.jit.load(path, map_location=device).eval()
            self.available = True
            log.info("ReID embedder loaded (%s, %s)", os.path.basename(path), device)
        except Exception:
            log.exception("could not load ReID checkpoint %s — continuing without it", path)

    def _forward(self, crop_bgr, size):
        torch = self.torch
        img = cv2.resize(crop_bgr, size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet normalisation, which every backbone in this family expects.
        rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / \
              np.array([0.229, 0.224, 0.225], np.float32)
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        v = out.flatten().float().cpu().numpy()
        n = np.linalg.norm(v)
        return None if n < 1e-8 else v / n

    def embed(self, crop_bgr):
        if not self.available:
            return None

        if self.size is not None:
            try:
                return self._forward(crop_bgr, self.size)
            except Exception:
                log.exception("embedding failed — dropping this crop's appearance")
                return None

        for size in self.INPUT_SIZES:
            try:
                v = self._forward(crop_bgr, size)
            except Exception:
                continue
            self.size = size
            log.info("ReID input size: %dx%d", *size)
            return v

        log.error("ReID model rejected every input size tried %s — "
                  "continuing on geography alone", list(self.INPUT_SIZES))
        self.available = False
        return None


# ---------------------------------------------------------------------------
# Tracks and identities
# ---------------------------------------------------------------------------

class Track:
    __slots__ = ("tid", "box", "entries", "crops", "last_frame", "best", "embeds", "gid")

    def __init__(self, tid, box, frame_idx):
        self.tid = tid
        self.box = box
        self.entries = []          # (frame, conf, sharpness, lat, lon)
        self.crops = {}            # frame -> BGR
        self.embeds = []
        self.last_frame = frame_idx
        self.best = (None, -1.0)   # (crop, sharpness)
        self.gid = None

    def add(self, frame_idx, box, conf, crop, lat, lon, embed=None):
        self.box = box
        self.last_frame = frame_idx
        s = sharpness(crop)
        self.entries.append((frame_idx, conf, s, lat, lon))
        self.crops[frame_idx] = crop
        if embed is not None:
            self.embeds.append(embed)
        if s > self.best[1]:
            self.best = (crop, s)

    @property
    def centroid(self):
        if not self.embeds:
            return None
        v = np.mean(self.embeds, axis=0)
        n = np.linalg.norm(v)
        return None if n < 1e-8 else v / n

    @property
    def position(self):
        pts = [(e[3], e[4]) for e in self.entries]
        return medoid(pts)


class Identity:
    """One casualty: every buffer ever assigned to them."""

    def __init__(self, gid):
        self.gid = gid
        self.embeds = []
        self.points = []           # every GPS fix, for the running medoid
        self.cumulative = 0
        self.first_frame = None
        self.buffers = 0

    @property
    def centroid(self):
        if not self.embeds:
            return None
        v = np.mean(self.embeds, axis=0)
        n = np.linalg.norm(v)
        return None if n < 1e-8 else v / n

    @property
    def position(self):
        return medoid(self.points)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

class MissionRun:
    def __init__(self, args, bus):
        self.args = args
        self.bus = bus
        self.writer = RunWriter(args.out)
        self.telem = TelemetryTrack(args.telem, args.telem_offset) if args.telem else None
        self.origin = tuple(float(v) for v in args.origin.split(","))
        self.geo = Geolocaliser(enabled=not args.no_geo,
                                zoom_fallback=getattr(args, "zoom", 1.0) or 1.0)
        # Detection runs on downscaled frames; the intrinsics are calibrated to
        # the sensor. Set once the first frame's true width is known.
        self.pixel_scale = 1.0
        self.geo_hits = 0
        self.geo_misses = 0

        self.tracks = []
        self.next_tid = 1
        self.identities = {}
        self.next_gid = 1

        self.frames_seen = 0
        self.detections = 0
        self.t_start = time.time()
        self._log_seq = 0
        self._frame_seq = 0
        self._last_stats = 0.0
        self._last_telem_t = None

    # -- emitting ----------------------------------------------------------

    def emit_log(self, level, msg):
        """One line to mission.log on disk and to the console, together."""
        self.writer.log(level, msg)
        self._log_seq += 1
        self.bus.emit("logs", [{"seq": self._log_seq, "ts": time.time(),
                                "level": level, "msg": msg}])

    # Emit rate in VIDEO seconds. Matching telem_worker's 4 Hz keeps the trail
    # the same density however the console was fed, and a 30 fps clip would
    # otherwise put 30 fixes a second on a line the map re-serialises each time.
    TELEM_HZ = 4.0

    def state_at(self, video_t, frame_idx=None):
        """
        Flight state for this moment, by frame number where the CSV gives one.

        A frame column is an exact mapping between the recording and the clip;
        falling back to timestamps means aligning two clocks that were never
        synchronised, which --telem-offset exists to patch up by hand.
        """
        if not self.telem:
            return None
        if frame_idx is not None:
            s = self.telem.at_frame(frame_idx)
            if s is not None:
                return s
        return self.telem.at(video_t)

    def emit_telemetry(self, video_t, frame_idx=None):
        if not self.telem:
            return
        if (self._last_telem_t is not None
                and video_t - self._last_telem_t < 1.0 / self.TELEM_HZ):
            return
        self._last_telem_t = video_t
        s = self.state_at(video_t, frame_idx)
        if s is None:
            return
        self.bus.emit("telemetry", {
            "uavId": 0,
            "latitude": s["lat"],
            "longitude": s["lon"],
            "altitude": s["alt"],
            "groundspeed": s["speed"],
            "battery": None,
            "batteryPct": s["battery"],
            "current": None,
            "armed": True,
            "mode": "AUTO",
            "heading": s["heading"],
            "Status": "ACTIVE",
            "Last_Heartbeat": 0.25,
            # Wall-clock, so the console's staleness check reads it as live even
            # when video time is running slower than real time.
            "ts": time.time(),
        })

    def emit_stats(self, video_t, force=False):
        now = time.time()
        if not force and now - self._last_stats < 0.5:
            return
        self._last_stats = now
        elapsed = now - self.t_start
        self.bus.emit("stats", {
            "frames": self.frames_seen,
            "gids": len(self.identities),
            "logLines": self._log_seq,
            "fpsLast": round(self.frames_seen / elapsed, 2) if elapsed > 0 else None,
            "fpsMin": None, "fpsMax": None,
            "levels": {},
            "elapsed": video_t,
            "paused": False,
            "speed": self.args.speed,
            "startedAt": self.t_start,
        })

    def emit_annotated(self, frame, video_t, t0):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return
        import base64
        self._frame_seq += 1
        self.bus.emit("frame", {
            "worker": "detect",
            "seq": self._frame_seq,
            "ts": time.time(),
            "jpeg": base64.b64encode(buf.tobytes()).decode("ascii"),
            "width": frame.shape[1],
            "height": frame.shape[0],
            "latencyMs": round((time.time() - t0) * 1000, 1),
            "backend": "yoloe",
            "device": self.args.device,
            "videoTime": round(video_t, 3),
        })

    def fix_at(self, video_t, frame_idx=None):
        """
        Where the drone was at this moment in the clip.

        With no CSV the coordinates are synthesised as a slow drift around
        --origin, exactly as ingest_video.py does: a GID needs *some* position
        to be written at all, and a drift keeps each track's medoid stable while
        still separating the ids. mission.log says they are synthesised, and the
        map shows no movement, so nothing downstream mistakes them for fixes.
        """
        if self.telem:
            s = self.state_at(video_t, frame_idx)
            return s["lat"], s["lon"]
        olat, olon = self.origin
        return (olat + 2e-5 * math.sin(video_t / 7.0),
                olon + 2e-5 * math.cos(video_t / 11.0))

    def locate(self, box, state, fix):
        """
        This detection's ground position, falling back to the drone's own fix.

        The fallback is what the console showed before projection existed, and
        it is still better than dropping the detection — but it collapses to
        "the casualty is under the drone", so it is counted and reported at the
        end of the run rather than passing silently.
        """
        pos = self.geo.locate(box, state, self.pixel_scale)
        if pos is not None:
            self.geo_hits += 1
            return pos
        self.geo_misses += 1
        return fix[0], fix[1]

    # -- identity assignment ----------------------------------------------

    def assign(self, track):
        """
        Decide which casualty a finished track belongs to.

        Appearance first, geography as a gate — the pipeline's own order. A
        track that looks like an existing GID *and* is lying within a few metres
        of it is that person coming back into frame; the same face forty metres
        away is someone else who happens to be dressed alike.
        """
        pos = track.position
        cent = track.centroid

        best_gid, best_score = None, 0.0
        for gid, ident in self.identities.items():
            ipos = ident.position
            dist = (haversine_m(pos[0], pos[1], ipos[0], ipos[1])
                    if pos and ipos else None)

            if dist is not None and dist > GEO_FAR_M:
                continue

            if cent is not None and ident.centroid is not None:
                score = float(np.dot(cent, ident.centroid))
                if score < self.args.reid_thresh:
                    continue
            elif dist is not None and dist <= GEO_MATCH_M:
                # No appearance to go on: accept only a very close fix, and say
                # so, because this is the weak path.
                score = 0.0
            else:
                continue

            if best_gid is None or score > best_score:
                best_gid, best_score = gid, score

        if best_gid is None:
            gid = self.next_gid
            self.next_gid += 1
            self.identities[gid] = Identity(gid)
            return gid, "new", 0.0
        return best_gid, "merge", best_score

    def flush_track(self, track):
        """Turn a finished track into a GID buffer: files, then a live event."""
        if len(track.entries) < self.args.min_hits:
            return
        pos = track.position
        if pos is None:
            return

        gid, event, score = self.assign(track)
        ident = self.identities[gid]

        ident.embeds.extend(track.embeds)
        ident.points.extend((e[3], e[4]) for e in track.entries)
        ident.cumulative += len(track.entries)
        ident.buffers += 1
        first = track.entries[0][0]
        ident.first_frame = first if ident.first_frame is None else min(ident.first_frame, first)

        med = ident.position
        fps = self.args.fps or 30.0

        if event == "merge":
            how = f"cosine={score:.3f}" if score else "geo match"
            self.emit_log("MATCH", f"tid={track.tid} -> gid={gid} ({how})")

        # Files first: the console resolves crop URLs against the run folder the
        # moment the gid event lands, so the images have to already be there.
        before = len(self.writer.lines)
        self.writer.write_buffer(
            gid=gid,
            event=event,
            tid=track.tid,
            entries=track.entries,
            crops=track.crops,
            lat=round(float(med[0]), 8),
            lon=round(float(med[1]), 8),
            cumulative=ident.cumulative,
            first_frame=ident.first_frame,
            ros_time=first / fps,
            representative=track.best[0],
        )
        # write_buffer logs through the writer only; mirror exactly the lines it
        # appended so the notification panel sees the same story as the file.
        for line in self.writer.lines[before:]:
            prefix, _, msg = line.partition("] ")
            self._log_seq += 1
            self.bus.emit("logs", [{"seq": self._log_seq, "ts": time.time(),
                                    "level": prefix.lstrip("[").strip(), "msg": msg}])
        self.writer.flush_log()

        payload = build_gid_payload(self.writer.run, gid)
        if payload:
            self.bus.emit("gid", payload)
            log.info("GID %d %s (tid=%d, gallery=%d) @ %.6f,%.6f",
                     gid, event, track.tid, ident.cumulative, med[0], med[1])

    # -- the loop ----------------------------------------------------------

    def run(self):
        args = self.args

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            log.error("cannot open %s", args.video)
            return 2

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        args.fps = fps

        # Projection works in the calibrated sensor's pixels, so a downscaled
        # working frame has to be scaled back up before it is projected.
        if args.width and src_w and src_w > args.width:
            self.pixel_scale = src_w / float(args.width)
        limit = min(total, args.max_frames) if args.max_frames else total

        log.info("video: %s — %d frames @ %.2f fps (%.1f s)%s",
                 os.path.basename(args.video), limit, fps, limit / fps,
                 f", projecting at {self.pixel_scale:.3f}x" if self.pixel_scale != 1.0 else "")

        from ultralytics import YOLOE
        log.info("loading YOLOE from %s", args.weights)
        model = YOLOE(args.weights)
        model.set_classes(DEFAULT_PROMPTS, model.get_text_pe(DEFAULT_PROMPTS))

        embedder = Embedder(args.reid, args.device)

        # mission_start is what resets every pane, so it goes out before any
        # frame does — and it names the run folder the relay serves crops from.
        self.bus.set_hello("mission_start", {
            "runFolder": self.writer.run,
            "name": os.path.basename(self.writer.run),
            "startedAt": datetime.now().isoformat(),
            "replay": False,
            "speed": args.speed,
        })
        self.bus.emit("mission_start", {
            "runFolder": self.writer.run,
            "name": os.path.basename(self.writer.run),
            "startedAt": datetime.now().isoformat(),
            "replay": False,
            "speed": args.speed,
        })

        self.emit_log("INFO", f"Mission Folder: {self.writer.run}")
        self.emit_log("OK", f"YOLOE detector loaded ({os.path.basename(args.weights)})")
        if embedder.available:
            self.emit_log("OK", f"ReID model loaded ({os.path.basename(args.reid)})")
        else:
            self.emit_log("WARN", "No ReID model — identities are matched on GPS proximity only")
        if self.geo.available:
            self.emit_log("OK", "Geolocalisation ready (pixel -> ground projection)")
        else:
            self.emit_log("WARN", f"No geolocalisation ({self.geo.reason}) — casualty "
                                  f"positions fall back to the drone's own fix")
        self.emit_log("INFO", f"Video source: {os.path.basename(args.video)}")
        if self.telem:
            self.emit_log("INFO", f"Telemetry source: {os.path.basename(args.telem)}")
        else:
            self.emit_log("WARN", "No telemetry CSV — GPS coordinates are SYNTHESISED "
                                  "and the drone will not move on the map")

        hub = None
        if args.hub_port:
            hub = FrameHub(args.hub_port)
            hub.start()

        wall0 = time.time()
        frame_idx = -1

        try:
            while frame_idx + 1 < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                self.frames_seen += 1
                video_t = frame_idx / fps
                t0 = time.time()

                if args.width and frame.shape[1] > args.width:
                    scale = args.width / frame.shape[1]
                    frame = cv2.resize(frame, (args.width, int(frame.shape[0] * scale)),
                                       interpolation=cv2.INTER_AREA)

                # Wall-clock pacing: skip whatever the processing could not keep
                # up with, so the demo stays in real time on a fast machine.
                if args.realtime:
                    behind = (time.time() - wall0) * args.speed - video_t
                    if behind > 2.0 / fps:
                        continue

                if hub:
                    hub.publish(frame)
                self.emit_telemetry(video_t, frame_idx)

                if frame_idx % args.stride == 0:
                    self.step(model, embedder, frame, frame_idx, video_t, t0)
                self.emit_stats(video_t)

                if not args.realtime and args.speed > 0:
                    # Video-time pacing. Only sleeps when the machine is running
                    # AHEAD of the clip; on CPU it never will, and the loop just
                    # runs flat out with telemetry still pinned to video time.
                    target = video_t / args.speed
                    slack = target - (time.time() - wall0)
                    if slack > 0:
                        time.sleep(min(slack, 0.25))

        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            cap.release()

        # Everything still open is a casualty the clip ended on.
        for t in self.tracks:
            self.flush_track(t)
        self.tracks = []

        self.emit_log("INFO", "=" * 40)
        self.emit_log("INFO", "MISSION COMPLETE")
        self.emit_log("INFO", f"Frames            : {self.frames_seen}")
        self.emit_log("INFO", f"GIDs created      : {len(self.identities)}")
        located = self.geo_hits + self.geo_misses
        if located:
            pct = 100.0 * self.geo_hits / located
            level = "INFO" if pct > 50 else "WARN"
            self.emit_log(level, f"Detections located: {self.geo_hits}/{located} "
                                 f"({pct:.0f}% projected to ground)")
        self.emit_log("INFO", f"Gallery entries   : {sum(i.cumulative for i in self.identities.values())}")
        self.emit_stats(self.frames_seen / fps, force=True)
        self.writer.flush_log()

        if hub:
            # Hold the hub open briefly so the workers drain rather than
            # flashing "source exhausted" the instant the clip ends.
            time.sleep(1.0)
            hub.stop()

        log.info("done — %d GIDs in %.1fs", len(self.identities), time.time() - self.t_start)
        print(self.writer.run)
        return 0

    def step(self, model, embedder, frame, frame_idx, video_t, t0):
        """One detection pass: detect, associate, annotate, publish."""
        args = self.args

        fix = self.fix_at(video_t, frame_idx)
        state = self.state_at(video_t, frame_idx)

        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        boxes = [tuple(float(v) for v in b) for b in res.boxes.xyxy.tolist()]
        confs = [float(c) for c in res.boxes.conf.tolist()]
        self.detections += len(boxes)

        self.emit_log("DET", f"frame={frame_idx} raw={len(boxes)} final={len(boxes)}")

        # Greedy IoU association, best pairs first.
        pairs = sorted(
            ((iou(t.box, b), ti, bi)
             for ti, t in enumerate(self.tracks) for bi, b in enumerate(boxes)),
            reverse=True,
        )
        used_t, used_b = set(), set()
        for score, ti, bi in pairs:
            if score < args.iou or ti in used_t or bi in used_b:
                continue
            crop = expand_crop(frame, boxes[bi])
            if crop is None:
                continue
            plat, plon = self.locate(boxes[bi], state, fix)
            self.tracks[ti].add(frame_idx, boxes[bi], confs[bi], crop,
                                plat, plon, embedder.embed(crop))
            used_t.add(ti)
            used_b.add(bi)
            self.emit_log("TRACK", f"tid={self.tracks[ti].tid} updated "
                                   f"len={len(self.tracks[ti].entries)}")

        for bi, b in enumerate(boxes):
            if bi in used_b:
                continue
            crop = expand_crop(frame, b)
            if crop is None:
                continue
            t = Track(self.next_tid, b, frame_idx)
            plat, plon = self.locate(b, state, fix)
            t.add(frame_idx, b, confs[bi], crop, plat, plon, embedder.embed(crop))
            self.tracks.append(t)
            self.emit_log("TRACK", f"tid={self.next_tid} created frame={frame_idx}")
            self.next_tid += 1

        # A track that has gone unseen long enough is flushed to a GID NOW,
        # rather than at the end of the clip. That is the whole point of this
        # script: the pin drops while the drone is still flying.
        gap = args.max_gap * args.stride
        still = []
        for t in self.tracks:
            if frame_idx - t.last_frame > gap:
                self.flush_track(t)
            else:
                still.append(t)
        self.tracks = still

        self.emit_annotated(self.annotate(frame, boxes, confs, video_t, fix), video_t, t0)

    def annotate(self, frame, boxes, confs, video_t, fix):
        """The Geolocalisation feed: boxes, ids, and where the drone was."""
        vis = frame.copy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = (int(v) for v in box)

            # Colour by the track this box belongs to, so a person keeps their
            # colour across frames and the eye can follow them.
            owner = next((t for t in self.tracks if iou(t.box, box) > 0.5), None)

            color = (80, 220, 120) if owner is None else self._tid_color(owner.tid)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"{conf:.2f}"
            if owner is not None:
                label = f"tid {owner.tid} · {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 7)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(vis, label, (x1 + 4, max(10, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, h - 46), (w, h), (0, 0, 0), -1)
        stamp = f"t+{int(video_t // 60):02d}:{video_t % 60:05.2f}"
        where = f"{fix[0]:.6f}, {fix[1]:.6f}"
        if not self.telem:
            where += "  (synthesised)"
        cv2.putText(vis, f"{stamp}   {where}", (8, h - 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis,
                    f"tracks {len(self.tracks)}   casualties {len(self.identities)}",
                    (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 255), 1,
                    cv2.LINE_AA)
        return vis

    @staticmethod
    def _tid_color(tid):
        """Same golden-angle hue walk the console uses for GID colours."""
        h = int((tid * 137.508) % 180)
        bgr = cv2.cvtColor(np.uint8([[[h, 180, 245]]]), cv2.COLOR_HSV2BGR)[0][0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def spawn_workers(args, source):
    """Bring up depth and traversability against the hub, and keep the handles."""
    here = os.path.dirname(os.path.abspath(__file__))
    workers = os.path.join(here, "workers")
    procs = []
    for name, backend, fps in (
        ("depth_worker.py", args.depth_backend, args.worker_fps),
        ("seg_worker.py", args.seg_backend, max(1.0, args.worker_fps - 1)),
    ):
        cmd = [sys.executable, os.path.join(workers, name),
               "--source", source, "--backend", backend, "--fps", str(fps)]
        log.info("starting %s (%s)", name, backend)
        procs.append(subprocess.Popen(cmd, cwd=workers))
    return procs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="the clip to fly")
    ap.add_argument("--telem", help="flight-controller CSV; without it the drone does not move")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="zoom ratio to assume for frames whose telemetry row "
                         "has no zoom_ratio; rows that do have one are used "
                         "as logged, per frame")
    ap.add_argument("--telem-offset", type=float, default=0.0,
                    help="seconds to shift the CSV against the video's start")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "demo_runs"))
    ap.add_argument("--weights", default=os.environ.get("YOLOE_WEIGHTS", "yoloe-26x-seg.pt"),
                    help="YOLOE weights; set YOLOE_WEIGHTS in Python/.env to override")
    ap.add_argument("--reid", default=os.environ.get("REID_MODEL"),
                    help="TorchScript ReID checkpoint (e.g. ../mobileclip2_b.ts)")
    ap.add_argument("--reid-thresh", type=float, default=0.62,
                    help="cosine similarity above which a track is the same casualty")
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--stride", type=int, default=5, help="detect every Nth frame")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.25, help="track association threshold")
    ap.add_argument("--max-gap", type=int, default=3,
                    help="detection passes a track may go unmatched before it becomes a GID")
    ap.add_argument("--min-hits", type=int, default=2,
                    help="detections a track needs before it counts as a casualty")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--width", type=int, default=960, help="downscale frames to this width")
    ap.add_argument("--no-geo", action="store_true",
                    help="skip pixel->ground projection; pin casualties at the drone's fix")
    ap.add_argument("--origin", default="28.7501,77.1177",
                    help="lat,lon used to synthesise GPS when --telem is absent")

    ap.add_argument("--speed", type=float, default=1.0, help="playback rate of video time")
    ap.add_argument("--realtime", action="store_true",
                    help="drop frames to hold wall-clock pace instead of processing every one")

    ap.add_argument("--hub-port", type=int, default=8090,
                    help="MJPEG port the model workers read; 0 disables the hub")
    ap.add_argument("--no-workers", action="store_true",
                    help="do not spawn depth/seg (run them yourself, or skip them)")
    ap.add_argument("--depth-backend", default=os.environ.get("DEPTH_BACKEND", "stub"))
    ap.add_argument("--seg-backend", default=os.environ.get("SEG_BACKEND", "stub"))
    ap.add_argument("--worker-fps", type=float, default=4.0)

    args = ap.parse_args()

    if not os.path.isfile(args.video):
        log.error("not a file: %s", args.video)
        return 64
    if args.telem and not os.path.isfile(args.telem):
        log.error("not a file: %s", args.telem)
        return 64

    # Both stacks must be in THIS interpreter. They are in different conda envs
    # on this machine, and the failure without this check is an ImportError
    # thrown after the model has already spent a minute loading.
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        log.error("ultralytics is not installed in %s", sys.executable)
        log.error("  %s -m pip install ultralytics", sys.executable)
        return 1
    try:
        import socketio  # noqa: F401
    except ImportError:
        log.error("python-socketio is not installed in %s", sys.executable)
        log.error("  %s -m pip install 'python-socketio[client]'", sys.executable)
        log.error("this script needs ultralytics AND python-socketio in ONE interpreter")
        return 1

    bus = Bus(role="mission_run")
    bus.connect()
    bus.wait_connected(timeout=5)

    run = MissionRun(args, bus)

    procs = []
    if not args.no_workers and args.hub_port:
        # The hub is not listening until run() starts it, but the workers retry
        # their source, so starting them here costs one failed connect apiece.
        procs = spawn_workers(args, f"http://127.0.0.1:{args.hub_port}/")

    try:
        return run.run()
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        bus.close()


if __name__ == "__main__":
    sys.exit(main())

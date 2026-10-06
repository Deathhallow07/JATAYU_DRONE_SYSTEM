#!/usr/bin/env python3
"""
Turn a video into a run folder the console can replay.

This is NOT the pipeline. It is the shortest honest path from "here is a video"
to "here are global IDs on screen", for when the full pipeline cannot run —
no ReID checkpoint, no CUDA, no telemetry, or simply no time before a demo.

What is real:
    - detections come from YOLOE on the actual video, with open-vocabulary
      prompts aimed at casualties
    - crops, confidences, frame numbers and sharpness are measured from those
      detections
    - tracks are formed by IoU association across sampled frames

What is not:
    - there is no ReID stage, so a person who leaves frame and returns becomes
      a NEW id. The pipeline's whole contribution is re-identifying them as the
      same GID; this cannot, and does not pretend to.
    - GPS is interpolated from --telem when given, and otherwise synthesised
      around --origin. Synthesised coordinates are marked in mission.log.

Anything it writes is in the pipeline's exact on-disk format (see run_writer),
so the watcher, the relay and the console cannot tell the difference — which is
the point: the console is exercised for real, on your footage.

    python ingest_video.py --video clip.mp4 --out ../demo_runs
    python ingest_video.py --video clip.mp4 --telem fcb.csv --stride 10
"""

import argparse
import csv
import logging
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_writer import RunWriter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ingest")

# Matches the pipeline's crop geometry so the cards look like the real thing.
CROP_W, CROP_H = 128, 256
BBOX_EXPAND = 1.5

DEFAULT_PROMPTS = [
    "person",
    "person lying on the ground",
    "injured person",
    "casualty",
]


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


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def load_telem(path):
    """
    Load a flight-controller CSV as [(t_mono_s, lat, lon), ...].

    Accepts the pipeline's column names and a few obvious aliases, because the
    recordings in the repo do not all agree on them.
    """
    t_keys = ("t_mono_s", "timestamp_sec", "t", "time")
    lat_keys = ("lat_deg", "drone_lat", "lat", "latitude")
    lon_keys = ("lon_deg", "drone_lon", "lon", "longitude")

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        pick = lambda keys: next((k for k in keys if k in reader.fieldnames), None)  # noqa: E731
        tk, lak, lok = pick(t_keys), pick(lat_keys), pick(lon_keys)
        if not (tk and lak and lok):
            raise ValueError(
                f"{path}: need a time + lat + lon column; found {reader.fieldnames}"
            )
        for r in reader:
            try:
                rows.append((float(r[tk]), float(r[lak]), float(r[lok])))
            except (TypeError, ValueError):
                continue

    rows.sort()
    log.info("telemetry: %d rows spanning %.1f s", len(rows), rows[-1][0] - rows[0][0] if rows else 0)
    return rows


def telem_at(rows, t):
    """Nearest-sample lookup. Good enough at the stride this script samples."""
    if not rows:
        return None
    i = min(range(len(rows)), key=lambda k: abs(rows[k][0] - t))
    return rows[i][1], rows[i][2]


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

class Track:
    __slots__ = ("tid", "box", "entries", "crops", "last_frame", "best")

    def __init__(self, tid, box, frame_idx):
        self.tid = tid
        self.box = box
        self.entries = []          # (frame, conf, sharpness, lat, lon)
        self.crops = {}            # frame -> BGR
        self.last_frame = frame_idx
        self.best = (None, -1.0)   # (crop, sharpness)

    def add(self, frame_idx, box, conf, crop, lat, lon):
        self.box = box
        self.last_frame = frame_idx
        s = sharpness(crop)
        self.entries.append((frame_idx, conf, s, lat, lon))
        self.crops[frame_idx] = crop
        if s > self.best[1]:
            self.best = (crop, s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "demo_runs"))
    ap.add_argument("--weights", default=os.environ.get("YOLOE_WEIGHTS", "yoloe-26x-seg.pt"),
                    help="YOLOE weights; set YOLOE_WEIGHTS in Python/.env to override")
    ap.add_argument("--telem", default=None, help="flight-controller CSV for real GPS")
    ap.add_argument("--stride", type=int, default=15, help="detect every Nth frame")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.25, help="track association threshold")
    ap.add_argument("--max-gap", type=int, default=3,
                    help="sampled frames a track may go unmatched before it closes")
    ap.add_argument("--min-hits", type=int, default=2,
                    help="detections a track needs before it becomes a GID")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N source frames")
    ap.add_argument("--origin", default="28.7501,77.1177",
                    help="lat,lon used to synthesise GPS when --telem is absent")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        log.error("cannot open %s", args.video)
        return 2

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    limit = min(total, args.max_frames) if args.max_frames else total

    telem = load_telem(args.telem) if args.telem else []
    olat, olon = (float(v) for v in args.origin.split(","))

    log.info("loading YOLOE from %s", args.weights)
    from ultralytics import YOLOE
    model = YOLOE(args.weights)
    model.set_classes(DEFAULT_PROMPTS, model.get_text_pe(DEFAULT_PROMPTS))

    writer = RunWriter(args.out)
    writer.log("INFO", f"Mission Folder: {writer.run}")
    writer.log("INFO", f"Source video: {args.video}")
    writer.log("OK", f"YOLOE detector loaded ({os.path.basename(args.weights)})")
    writer.log("INFO", f"Prompts: {', '.join(DEFAULT_PROMPTS)}")
    if telem:
        writer.log("INFO", f"Telemetry: {args.telem}")
    else:
        writer.log("WARN", "No telemetry CSV — GPS coordinates are SYNTHESISED")
    writer.log("WARN", "Ingest mode: detection + IoU tracking only, no ReID stage")

    active, closed = [], []
    next_tid = 1
    t0 = time.time()
    processed = 0

    for frame_idx in range(0, limit, args.stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        processed += 1

        t_sec = frame_idx / fps
        fix = telem_at(telem, t_sec)
        if fix is None:
            # A slow drift, so the medoid of a track's fixes is stable but the
            # ids do not all land on one point.
            fix = (olat + 2e-5 * math.sin(t_sec / 7.0), olon + 2e-5 * math.cos(t_sec / 11.0))

        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        boxes = [tuple(float(v) for v in b) for b in res.boxes.xyxy.tolist()]
        confs = [float(c) for c in res.boxes.conf.tolist()]

        writer.log("FRAME", f"FPS={processed / max(1e-6, time.time() - t0):.1f}")
        writer.log("DET", f"frame={frame_idx} raw={len(boxes)} final={len(boxes)}")

        # Greedy IoU association, best pairs first.
        pairs = sorted(
            ((iou(t.box, b), ti, bi) for ti, t in enumerate(active) for bi, b in enumerate(boxes)),
            reverse=True,
        )
        used_t, used_b = set(), set()
        for score, ti, bi in pairs:
            if score < args.iou or ti in used_t or bi in used_b:
                continue
            crop = expand_crop(frame, boxes[bi])
            if crop is None:
                continue
            active[ti].add(frame_idx, boxes[bi], confs[bi], crop, *fix)
            used_t.add(ti)
            used_b.add(bi)
            writer.log("TRACK", f"tid={active[ti].tid} updated len={len(active[ti].entries)}")

        for bi, b in enumerate(boxes):
            if bi in used_b:
                continue
            crop = expand_crop(frame, b)
            if crop is None:
                continue
            t = Track(next_tid, b, frame_idx)
            t.add(frame_idx, b, confs[bi], crop, *fix)
            active.append(t)
            writer.log("TRACK", f"tid={next_tid} created frame={frame_idx}")
            next_tid += 1

        still, gap = [], args.max_gap * args.stride
        for t in active:
            (still if frame_idx - t.last_frame <= gap else closed).append(t)
        active = still

        if processed % 10 == 0:
            log.info("frame %d/%d  tracks active=%d closed=%d  (%.1fs)",
                     frame_idx, limit, len(active), len(closed), time.time() - t0)

    cap.release()
    closed.extend(active)

    # Only tracks seen more than once become identities — a single frame is as
    # likely to be a false positive as a person.
    gids = [t for t in closed if len(t.entries) >= args.min_hits]
    gids.sort(key=lambda t: t.entries[0][0])

    for gid, t in enumerate(gids, start=1):
        lats = [e[3] for e in t.entries]
        lons = [e[4] for e in t.entries]
        writer.write_buffer(
            gid=gid,
            event="new",
            tid=t.tid,
            entries=t.entries,
            crops=t.crops,
            lat=round(float(np.median(lats)), 8),
            lon=round(float(np.median(lons)), 8),
            cumulative=len(t.entries),
            first_frame=t.entries[0][0],
            ros_time=t.entries[0][0] / fps,
            representative=t.best[0],
        )

    writer.log("INFO", "=" * 40)
    writer.log("INFO", "INGEST COMPLETE")
    writer.log("INFO", f"Frames sampled       : {processed}")
    writer.log("INFO", f"Tracks formed        : {len(closed)}")
    writer.log("INFO", f"GIDs created         : {len(gids)}")
    writer.log("INFO", f"Gallery entries      : {sum(len(t.entries) for t in gids)}")
    writer.flush_log()

    log.info("done in %.1fs — %d GIDs", time.time() - t0, len(gids))
    print(writer.run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

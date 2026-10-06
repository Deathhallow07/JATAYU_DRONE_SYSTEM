# =========================================================
# MODULE 1: IMPORTS + CONFIGURATION
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import time
import json
import onnxruntime as ort
import os
from ultralytics import YOLOE
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # bundled: pipeline/ ships its own modules
from coordinate_transformer import CoordinateTransformer
from sklearn.cluster import DBSCAN


#sys.path.append('/media/uasdtu/DataSets1/SINet+Forestpersons/Pipeline_trial_3')
#from TD_Resnet_v3 import TD_2_NETWORK_V8

#sys.path.append('/media/uasdtu/DataSets1/SINet+Forestpersons')
#from Network_Detector import Network


# =========================================================
#                    CONFIGURATION
# =========================================================

RUN_FOLDER = None

MISSION_STATS = {

    "frames":0,

    "detections_raw":0,
    "detections_final":0,

    "blur_rejects":0,
    "shadow_rejects":0,
    "fov_rejects":0,
    "invalid_rejects":0,
    "gps_failures":0,

    "tracks_created":0,
    "tracks_updated":0,
    "tracks_deleted":0,

    "gids_created":0,
    "gallery_merges":0,

    "fps_sum":0.0,
    "fps_min":1e9,
    "fps_max":0.0,

    "start_time":time.time()
}

# -------- FILE PATHS --------
# VIDEO_PATH        = "/media/uasdtu/DataSets1/first_morning_flight/rosbag2_2026_04_09-18_28_10/cropped-output.mp4"
#VIDEO_PATH      = "/media/uasdtu/DataSets1/tanmay/harsh-lying-down-cropped.mp4"

#SINET_MODEL_PATH  = "/media/uasdtu/DataSets1/darsh/Pipeline/SINet+TDReID/sinet_detector_epoch_008.pth"
#REID_CHECKPOINT   = "/media/uasdtu/DataSets1/darsh/Pipeline/SINet+TDReID/epoch_003.pth"

# Accepts .pt or exported .onnx / .engine.
#
# An .onnx MUST be exported with YOLOE_CLASSES already applied,
# otherwise the prompts are not in the graph:
#
#     m = YOLOE("yoloe-11s-seg.pt")
#     m.set_classes(YOLOE_CLASSES, m.get_text_pe(YOLOE_CLASSES))
#     m.export(format="onnx", imgsz=YOLOE_IMGSZ)
#
# The export imgsz is baked in too, so it must match YOLOE_IMGSZ
# (or export with dynamic=True).



USE_ROBUST_DETECT = True
# YOLOE_WEIGHTS  = "/home/akshit/geolocalization/models/yoloe-26x-seg.pt"
YOLOE_WEIGHTS  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yoloe-26s-seg.pt")
REID_ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epoch_003.pth")
# REID_ONNX_PATH = "/home/uas/deployment_gate_1/scripts/pipeline/yoloe/epoch_003.pth"

# -------- GALLERY DUMP (continuous, per buffer flush) --------
SAVE_GALLERIES   = False
GALLERY_SAVE_DIR = "/home/akshit/geolocalization/galleries"  # set at runtime inside RUN_FOLDER

# -------- TELEMETRY SOURCE --------
# "live" -> Giver (ROS)
# "csv"  -> replay from CSV
TELEM_SOURCE   = "csv"
TELEM_CSV_PATH = "/home/uas/localisation/yoloe/videos/fcb_tmux_bags/2xzoom1ms20m/fcb-20260912-085849.csv"
# TELEM_CSV_PATH = "/home/uas/fcb_recordings/fcb-20260910-160213.csv"
# Fallback zoom ratio, used ONLY for frames the telemetry cannot speak for:
# a CSV with no zoom_ratio column, a blank cell, or the live Giver, which does
# not publish zoom. When the CSV does carry zoom_ratio it is read per frame and
# this constant is never consulted -- a survey clip that zooms mid-flight
# projects correctly either way.
#
# It is a RATIO relative to the zoom CAMERA_INTRINSICS were calibrated at, so
# 1.0 means "the intrinsics as written".
ZOOM_RATIO = 2.0

    
# -------- VIDEO SOURCE --------
# "live"  -> Giver (ROS camera + telem)
# "video" -> local file (pairs with TELEM_SOURCE for telem)
VIDEO_SOURCE     = "video"
START_FRAME = 7500
VIDEO_PATH       = "/home/uas/localisation/yoloe/videos/fcb_tmux_bags/2xzoom1ms20m/fcb-20260912-085849.mp4"
# VIDEO_PATH       = "/home/uas/fcb_recordings/fcb-20260910-160213.avi"

# -------- OUTPUT DISPLAY --------
# How you want to watch the annotated output.
#
#   "window"  cv2.imshow pop-up on THIS machine. Needs a display -- over
#             SSH that means `ssh -X`/`ssh -Y` -- and an opencv build
#             with GUI support (opencv-python, NOT opencv-python-headless,
#             which raises "The function is not implemented"). ESC in the
#             window ends the run.
#
#   "stream"  MJPEG over HTTP at http://<this-host>:STREAM_PORT/ , for
#             when the pipeline runs on a headless box. Open it in any
#             browser or with VLC. Note this is plain HTTP, not HTTPS,
#             and it binds 0.0.0.0 -- anyone who can reach the port sees
#             the feed, so keep it on a trusted network./
#
#   "none"    draw nothing anywhere. Fastest option, and the one to use
#             for an unattended run; mission artefacts are still written
#             to RUN_FOLDER exactly the same.
DISPLAY_MODE = "stream"
STREAM_PORT  = 8080

if DISPLAY_MODE not in ("window", "stream", "none"):
    raise ValueError(
        f"DISPLAY_MODE must be 'window', 'stream' or 'none', "
        f"got {DISPLAY_MODE!r}"
    )

VIDEO_START_UNIX = None               # ROS time at first video frame
TARGET_FPS       = None               # None = native FPS

# -------- START FRAME --------
# Frame to begin processing from, so a long clip can be entered part-way
# through without cutting it first. 0 starts at the beginning.
#
# Counted from the START OF THE VIDEO, which is deliberately the same
# numbering the telemetry lookup uses: read_source() turns the absolute
# frame index into t_mono_s as frame_idx / fps, so seeking the video and
# seeking the CSV are one operation -- set this and the telemetry
# follows by itself. VIDEO_START_UNIX stays anchored to video t=0 and is
# NOT shifted, because it is the wall-clock of the first frame of the
# FILE, not of wherever this run happens to start.






# CSV columns expected:
# timestamp_sec, drone_lat, drone_lon, drone_altitude_agl,
# roll_deg, pitch_deg, yaw_deg

# -------- CAMERA INTRINSICS --------
# CAMERA_INTRINSICS = np.array([                    ####  D455
#     [393.12779563, 0.0, 321.48263787],
#     [0.0, 394.76440916, 241.58044155],
#     [0.0, 0.0, 1.0]
# ])

# CAMERA_INTRINSICS = np.array([                      #### D415
#     [624.418806, 0.0, 326.10497],
#     [0.0, 625.47820365, 236.134524],
#     [0.0, 0.0, 1.0]
# ])

CAMERA_INTRINSICS = np.array([                          ####  SONY FCB
    [1584.02798,    0,        953.27546],
    [   0,       1581.52473,  518.97355],
    [   0,          0,          1       ]
])


transformer = CoordinateTransformer(CAMERA_INTRINSICS)

if VIDEO_SOURCE == "live":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # bundled: pipeline/ ships its own modules
    from giver_node import Giver
    giver = Giver()
else:
    giver = None


_CSV_TELEM = None  # numpy array, sorted by timestamp
_CSV_HAS_ZOOM = False  # True once a CSV with a usable zoom_ratio column is loaded

def load_csv_telemetry(path):
    """
    Loads flight-controller CSV with columns:
      frame, t_wall_utc, t_mono_s, lat_deg, lon_deg, alt_msl_m, alt_rel_m,
      gps_age_ms, roll_deg, pitch_deg, yaw_deg, att_age_ms, heading_deg,
      groundspeed_ms, hud_age_ms, zoom_ratio, zoom_position, zoom_age_ms,
      fc_time_boot_ms
    """
    global _CSV_TELEM, _CSV_HAS_ZOOM
    import csv
    from datetime import datetime as _dt

    # Columns this loader reads, in the order they are stored.
    NEEDED = ["t_mono_s", "lat_deg", "lon_deg", "alt_rel_m",
              "roll_deg", "pitch_deg", "yaw_deg"]

    # Read when present, never required. zoom_ratio is the camera's optical
    # zoom for that row; it scales the focal length, so it has to travel with
    # the frame rather than being fixed for the run. A log without the column
    # (an older recording, or the other CSV schema) falls back to ZOOM_RATIO,
    # which is why a missing column is not an error. Stored as NaN per row when
    # unavailable so the fallback is decided at lookup, per frame, not here.
    OPTIONAL = ["zoom_ratio"]
    STORED = NEEDED + OPTIONAL

    def _num(v):
        """
        Parse one CSV cell, or return None if it holds no number.

        A flight-controller log is written at a fixed rate from
        whatever MAVLink has delivered so far, so a cell is routinely
        EMPTY rather than absent: rows recorded before the first GPS
        fix carry a blank lat_deg/lon_deg, and a recording killed
        mid-write leaves a final row of bare commas. csv.DictReader
        hands those over as "" (or None for a short row), and float("")
        is the ValueError this used to die on.
        """
        if v is None:
            return None
        v = v.strip()
        if v == "" or v.lower() in ("nan", "none", "null"):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    rows = []
    skipped = 0
    first_bad = None

    # encoding / errors / newline are all deliberate:
    #
    #   utf-8-sig   strips a UTF-8 BOM if the log was written on Windows
    #               or by a tool that emits one. Left in place the BOM
    #               becomes part of the FIRST column name, so "frame"
    #               reads back as "\ufefframe" and the schema check below
    #               reports a missing column that is plainly present.
    #
    #   errors=     a flight-controller log is not guaranteed clean
    #   "replace"   UTF-8: a dropped serial byte, a half-written row from
    #               a recording that was killed, or a latin-1 degree sign
    #               in a comment field all put a byte in the stream that
    #               strict UTF-8 refuses. That was raising at
    #               reader.fieldnames -- and note the failure surfaces
    #               there even when the bad byte is far below the header,
    #               because the first read decodes a whole ~8 KB buffer.
    #               "replace" is used rather than "ignore" on purpose:
    #               it substitutes U+FFFD, which _num() cannot parse, so
    #               the damaged row is SKIPPED. "ignore" would delete the
    #               byte instead and silently turn "28.7\xff5" into a
    #               perfectly valid, perfectly wrong 28.75.
    #
    #   newline=""  what the csv module documents it wants, so quoted
    #               fields containing newlines are not mangled.
    with open(path, "r", encoding="utf-8-sig",
              errors="replace", newline="") as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError(f"Telemetry CSV is empty: {path}")

        # Header cells still get whitespace-stripped: a "lat_deg, lon_deg"
        # style header with spaces after the commas is otherwise a
        # missing-column error too.
        reader.fieldnames = [(c or "").strip() for c in reader.fieldnames]

        missing = [c for c in NEEDED if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(
                f"Telemetry CSV {path} is missing column(s) {missing}.\n"
                f"  found  : {list(reader.fieldnames)}\n"
                f"  expected flight-controller schema: {NEEDED}\n"
                f"A CSV headed timestamp_sec/drone_lat/drone_lon/... is "
                f"the OTHER schema and is not read by this loader."
            )

        for line_no, r in enumerate(reader, start=2):   # 1 = header

            # ---- Option A: use monotonic seconds (matches video elapsed time)
            # ---- Option B: use wall-clock UTC as Unix timestamp
            # ts_wall = _dt.fromisoformat(r["t_wall_utc"]).timestamp()

            vals = [_num(r.get(c)) for c in NEEDED]

            # A blank or absent zoom_ratio is NaN, not a dropped row: the
            # projection still works off the fallback, and losing the whole
            # row would lose a perfectly good fix with it.
            opt = [_num(r.get(c)) for c in OPTIONAL]
            opt = [float("nan") if v is None else v for v in opt]

            # A row is only usable if EVERY field parsed. Partial rows are
            # dropped rather than zero-filled on purpose: get_csv_telem()
            # is a nearest-neighbour lookup with no validity flag, so a
            # zero-filled lat/lon would be handed to bbox_to_gps() as a
            # real fix at (0, 0) and silently geolocate casualties into
            # the Gulf of Guinea. Losing the row is recoverable; a
            # plausible-looking wrong coordinate is not.
            if any(v is None for v in vals):
                skipped += 1
                if first_bad is None:
                    bad = [c for c, v in zip(NEEDED, vals) if v is None]
                    first_bad = (line_no, bad)
                continue

            rows.append(tuple(vals) + tuple(opt))

    if skipped:
        log_warn(
            f"Skipped {skipped} of {skipped + len(rows)} telemetry rows "
            f"with blank/unparseable fields (first at CSV line "
            f"{first_bad[0]}, empty: {first_bad[1]})"
        )

    if not rows:
        raise RuntimeError(
            f"No usable telemetry rows in {path} - all "
            f"{skipped} row(s) had blank or unparseable fields."
        )

    rows.sort(key=lambda x: x[0])
    _CSV_TELEM = np.array(rows, dtype=np.float64)
    log_ok(f"Loaded {len(rows)} CSV telemetry rows from {path}")

    # Report which zoom the run will actually project with, because a silent
    # fallback to ZOOM_RATIO on a clip shot at a different zoom moves every
    # pin by the ratio between them.
    zoom_col = _CSV_TELEM[:, len(NEEDED)]
    known = np.isfinite(zoom_col) & (zoom_col > 0)
    _CSV_HAS_ZOOM = bool(known.any())
    if _CSV_HAS_ZOOM:
        blanks = int((~known).sum())

        # Gaps are filled from the nearest row that DOES have a zoom, not from
        # ZOOM_RATIO. zoom_ratio is a MAVLink message like any other: it
        # arrives at its own rate and is simply absent for the first second or
        # two of a log, and across any dropout after. The lens did not move
        # during those rows, so the last (or first) value the camera reported
        # is the right one -- and it beats a constant that may describe a
        # different clip entirely. Forward-fill, then back-fill the head.
        if blanks:
            idx = np.where(known, np.arange(len(zoom_col)), -1)
            np.maximum.accumulate(idx, out=idx)
            head = idx < 0                      # rows before the first report
            idx[head] = int(np.argmax(known))   # ... take the first one
            _CSV_TELEM[:, len(NEEDED)] = zoom_col[idx]
            zoom_col = _CSV_TELEM[:, len(NEEDED)]

        log_ok(
            f"[ZOOM] per-frame zoom_ratio from CSV: "
            f"{zoom_col.min():.2f}x - {zoom_col.max():.2f}x"
            + (f" ({blanks} row(s) had none and took the nearest reported "
               f"value)" if blanks else "")
        )
    else:
        log_warn(
            f"[ZOOM] CSV carries no usable zoom_ratio - projecting every "
            f"frame at the fallback ZOOM_RATIO={ZOOM_RATIO:.2f}x"
        )


def get_csv_telem(timestamp_sec):
    """Nearest-neighbour lookup. Returns telem dict matching Giver format."""
    if _CSV_TELEM is None or len(_CSV_TELEM) == 0:
        return None
    idx = int(np.argmin(np.abs(_CSV_TELEM[:, 0] - timestamp_sec)))
    row = _CSV_TELEM[idx]

    # Column 7 is zoom_ratio, NaN on any row the log did not fill in.
    zoom = float(row[7]) if row.shape[0] > 7 else float("nan")
    if not np.isfinite(zoom) or zoom <= 0:
        zoom = None          # telem_zoom() applies the fallback

    return {
        "timestamp_sec": float(row[0]),
        "latitude":      float(row[1]),
        "longitude":     float(row[2]),
        "altitude":      float(row[3]),
        "roll":          float(np.radians(row[4])),
        "pitch":         float(np.radians(row[5])),
        "yaw":           float(np.radians(row[6])),
        "zoom":          zoom,
    }


def telem_zoom(telem):
    """
    Zoom ratio to project THIS frame with.

    Optical zoom multiplies the focal length, so fx/fy -- and with them the
    metres-per-pixel of every detection -- change whenever the operator zooms.
    Taking it from the telemetry row keeps the intrinsics following the camera
    through a clip that zooms part-way; ZOOM_RATIO is only the answer for
    frames the telemetry cannot speak for (live Giver, blank cell, older CSV
    schema with no zoom_ratio column).
    """
    z = telem.get("zoom") if telem else None
    if z is None:
        z = ZOOM_RATIO
    try:
        z = float(z)
    except (TypeError, ValueError):
        return float(ZOOM_RATIO)
    if not np.isfinite(z) or z <= 0:
        return float(ZOOM_RATIO)
    return z


# =========================================================
# UNIFIED VIDEO SOURCE
# =========================================================

_cap        = None
_video_fps  = None
_frame_time = None

# Frame index the capture actually landed on after seeking to
# START_FRAME. __main__ seeds frame_id from this, NOT from START_FRAME.
_start_frame_actual = 0

def init_video_source():
    global _cap, _video_fps, _frame_time, _start_frame_actual
    if VIDEO_SOURCE == "video":
        _cap = cv2.VideoCapture(VIDEO_PATH)
        if not _cap.isOpened():
            raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
        _video_fps  = _cap.get(cv2.CAP_PROP_FPS)
        fps_target  = TARGET_FPS if TARGET_FPS else _video_fps
        _frame_time = 1.0 / fps_target
        log_info(f"[VIDEO] file={VIDEO_PATH} native_fps={_video_fps} target_fps={fps_target}")

        _start_frame_actual = 0

        if START_FRAME > 0:

            total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if total > 0 and START_FRAME >= total:
                raise RuntimeError(
                    f"START_FRAME={START_FRAME} is past the end of "
                    f"{VIDEO_PATH} ({total} frames)"
                )

            _cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

            # Not every codec can seek to an arbitrary frame -- many land
            # on the nearest preceding keyframe instead. Take the
            # position the container ACTUALLY reports rather than
            # assuming it obeyed: frame_id is what drives the CSV
            # lookup, so believing a seek that did not happen would
            # desynchronise the telemetry by the seek error for the
            # whole run, silently.
            _start_frame_actual = int(_cap.get(cv2.CAP_PROP_POS_FRAMES))

            log_info(
                f"[VIDEO] START_FRAME={START_FRAME} "
                f"(landed on {_start_frame_actual}, "
                f"t_mono_s={(_start_frame_actual + 1) / _video_fps:.3f}s"
                + (f", of {total} frames)" if total else ")")
            )
    elif VIDEO_SOURCE == "live":
        log_info("[VIDEO] Live Giver stream")
    else:
        raise ValueError(f"Invalid VIDEO_SOURCE: {VIDEO_SOURCE!r}")

def read_source(frame_idx):
    global VIDEO_START_UNIX  # allow __main__ assignment to be visible here
    """
    Unified reader. Returns (ret, frame, telem).
    telem matches Giver dict shape (may be None).
    """
    if VIDEO_SOURCE == "live":
        return giver.read()

    ret, frame = _cap.read()
    if not ret:
        return False, None, None

    ts_elapsed = frame_idx / _video_fps          # matches t_mono_s in CSV
    ts_wall    = VIDEO_START_UNIX + ts_elapsed   # keeps downstream wall-clock

    if TELEM_SOURCE == "csv" and USE_GPS:
        telem = get_csv_telem(ts_elapsed)
    else:
        telem = None

    if telem is not None:
        telem["timestamp_sec"] = ts_wall   # downstream uses wall time

    return True, frame, telem


def bbox_to_gps(x1, y1, x2, y2, telem):
    """
    Converts bbox to GPS using the telem snapshot captured
    at the same frame — guarantees per-frame sync.
    telem is a dict with keys: lat, lon, alt, roll, pitch, yaw.
    """

    if telem is None or telem.get("latitude") is None:
        return None, None

    width  = x2 - x1
    height = y2 - y1

    # Geolocalise the bbox CENTRE -- the pixel->GPS transform is
    # calibrated for the centre point, not the feet / top-left.
    cx = x1 + 0.5 * width
    cy = y1 + 0.5 * height

    result = transformer.pixel_to_gps(
        u=cx,
        v=cy,
        drone_lat=telem["latitude"],
        drone_lon=telem["longitude"],
        drone_altitude=telem["altitude"],
        roll=telem["roll"],
        pitch=telem["pitch"],
        yaw=telem["yaw"],
        ground_altitude=0.0,
        # Per-frame, from the CSV's zoom_ratio where it has one. The
        # transformer scales fx/fy by this, so the ray for a given pixel
        # narrows exactly as the lens does.
        zoom=telem_zoom(telem),
    )

    if result is None:
        return None, None

    return result


def save_new_gid_artifacts(gid, gallery, assignment_ros_time,
                            event="new", tid=None, buffer_gallery=None):
    """
    Save/append artifacts for a Global ID.

    Called both when a GID is first created AND every time an existing
    GID is updated with a merged buffer, so metadata.txt accumulates the
    full history of buffers assigned to this GID (one appended record per
    call) instead of only the very first one. That history -- each
    buffer's entries plus the GID's medoid at that point in time -- lets
    you trace exactly when/why a track was merged into an existing GID
    versus spawning a new one.

    gallery        : REID_DICT[gid]["gallery"] AFTER this update (the
                     full/cumulative gallery representing the GID right now).
    buffer_gallery : just the entries this buffer contributed (defaults to
                     `gallery` for the "new" event, where they're the same).

    Compatible with current Pipeline.py list-based gallery format:
    [emb, conf, frame_id, cx, cy, s, sharpness, lat, lon, crop_img]
    """

    global RUN_FOLDER, GID_GPS_HISTORY

    if RUN_FOLDER is None:
        log_err(
            f"RUN_FOLDER is None. Cannot save artifacts for GID={gid}."
        )
        return

    if buffer_gallery is None:
        buffer_gallery = gallery

    gid_folder = os.path.join(RUN_FOLDER, f"gid_{gid}")
    os.makedirs(gid_folder, exist_ok=True)

    # ------------------------------------------------------
    # Representative crop (sharpest image)
    # ------------------------------------------------------

    best_item = max(gallery, key=lambda x: x[6])

    rep_crop = best_item[9]

    if rep_crop is not None and rep_crop.size > 0:

        rep_path = os.path.join(
            gid_folder,
            "representative.jpg"
        )

        if cv2.imwrite(rep_path, rep_crop):

            log_ok(
                f"GID={gid} Representative saved "
                f"(frame={best_item[2]} sharpness={best_item[6]:.1f})"
            )

        else:

            log_warn(
                f"GID={gid} Failed to save representative crop."
            )

    else:

        log_warn(
            f"GID={gid} Representative crop missing."
        )

    # ------------------------------------------------------
    # Save crops contributed by THIS buffer only (the full
    # cumulative gallery's crops are already handled by
    # save_gallery_crops on every 'new'/'merge' event).
    # ------------------------------------------------------

    crops_saved = 0

    for item in buffer_gallery:

        crop = item[9]

        if crop is None or crop.size == 0:
            continue

        fname = f"crop_frame_{item[2]:06d}.jpg"

        if cv2.imwrite(
            os.path.join(gid_folder, fname),
            crop
        ):
            crops_saved += 1

    log_ok(
        f"GID={gid} Saved {crops_saved}/{len(buffer_gallery)} buffer crops."
    )

    # ------------------------------------------------------
    # GPS: this buffer's points + the GID's running medoid
    # (computed over the FULL cumulative gallery, i.e. the
    # GID's position estimate as of this update)
    # ------------------------------------------------------

    buffer_gps_points = [
        (item[7], item[8])
        for item in buffer_gallery
        if item[7] is not None and item[8] is not None
    ]

    full_gps_points = [
        (item[7], item[8])
        for item in gallery
        if item[7] is not None and item[8] is not None
    ]

    if full_gps_points:
        medoid_lat, medoid_lon = robust_gallery_medoid(full_gps_points)
    else:
        medoid_lat, medoid_lon = None, None

    # ------------------------------------------------------
    # FINAL GPS: DBSCAN medoid over every frame from every
    # buffer EVER assigned to this GID (new + all merges),
    # including ones since pruned out of the cumulative
    # gallery by the M/N/L cap. This is the GID's one true
    # "final" coordinate, refreshed on every buffer.
    # ------------------------------------------------------

    gid_history = GID_GPS_HISTORY.setdefault(gid, [])
    gid_history.extend(buffer_gps_points)

    if gid_history:
        final_lat, final_lon = robust_gallery_medoid(gid_history)
    else:
        final_lat, final_lon = None, None

    # ------------------------------------------------------
    # Statistics (over the full cumulative gallery)
    # ------------------------------------------------------

    best_conf = max(item[1] for item in gallery)

    best_sharpness = max(item[6] for item in gallery)

    first_frame = min(item[2] for item in gallery)

    last_frame = max(item[2] for item in gallery)

    global MISSION_START_ROS_TIME

    if (assignment_ros_time is None or MISSION_START_ROS_TIME is None):
        mission_elapsed = 0.0
    else:
        mission_elapsed = (assignment_ros_time - MISSION_START_ROS_TIME)

    hrs = int(mission_elapsed // 3600)
    mins = int((mission_elapsed % 3600) // 60)
    secs = mission_elapsed % 60

    # ------------------------------------------------------
    # Metadata -- header (identity + Final GPS Coordinate)
    # written once at file creation, then REWRITTEN in place
    # on every subsequent call so "Final GPS" always reflects
    # every buffer to date. Buffer records below it are pure
    # append-only, so that history is never touched.
    # ------------------------------------------------------

    meta_path = os.path.join(
        gid_folder,
        "metadata.txt"
    )

    FINAL_GPS_START = "---- FINAL GPS (DBSCAN medoid, all buffers to date) ----\n"
    FINAL_GPS_END   = "---------------------------------------------------------\n"

    final_gps_block = (
        FINAL_GPS_START
        + f"Final GPS Samples   : {len(gid_history)}\n"
        + f"Final GPS Latitude  : {final_lat}\n"
        + f"Final GPS Longitude : {final_lon}\n"
        + FINAL_GPS_END
    )

    if not os.path.exists(meta_path):

        with open(meta_path, "w") as f:
            f.write("====================================================\n")
            f.write("GLOBAL IDENTITY RECORD\n")
            f.write("====================================================\n\n")

            f.write(f"GID                 : {gid}\n")
            f.write(f"Created             : {datetime.now().isoformat()}\n")
            f.write(f"Mission Folder      : {RUN_FOLDER}\n\n")

            f.write(final_gps_block)
            f.write("\n")

    else:

        with open(meta_path, "r") as f:
            content = f.read()

        start_idx = content.find(FINAL_GPS_START)

        if start_idx == -1:
            # Older metadata.txt from before this block existed --
            # drop it in right after the header instead of failing.
            insert_at = content.find("\n\n") + 2
            content = (
                content[:insert_at]
                + final_gps_block + "\n"
                + content[insert_at:]
            )
        else:
            end_idx = content.find(FINAL_GPS_END, start_idx) + len(FINAL_GPS_END)
            content = content[:start_idx] + final_gps_block + content[end_idx:]

        with open(meta_path, "w") as f:
            f.write(content)

    with open(meta_path, "a") as f:

        f.write("----------------------------------------------------\n")
        f.write(f"BUFFER {event.upper()}   TID={tid}\n")
        f.write("----------------------------------------------------\n")

        f.write(f"Wall Time           : {datetime.now().isoformat()}\n")

        if assignment_ros_time is not None:
            f.write(f"ROS Assignment Time : {assignment_ros_time:.3f} sec\n")
        else:
            f.write(f"ROS Assignment Time : N/A (no telemetry)\n")

        f.write(
            f"Mission Timestamp   : "
            f"{hrs:02d}:{mins:02d}:{secs:06.3f}\n\n"
        )

        f.write("------------ Gallery Summary (cumulative) ------------\n")

        f.write(f"Buffer Gallery Size : {len(buffer_gallery)}\n")
        f.write(f"Cumulative Gallery  : {len(gallery)}\n")
        f.write(f"Buffer Crops Saved  : {crops_saved}\n")
        f.write(f"First Frame         : {first_frame}\n")
        f.write(f"Last Frame          : {last_frame}\n")
        f.write(f"Best Confidence     : {best_conf:.4f}\n")
        f.write(f"Best Sharpness      : {best_sharpness:.2f}\n")
        f.write(f"Representative Frame: {best_item[2]}\n\n")

        f.write("------------ GID Medoid @ this update ------------\n")

        f.write(f"GPS Samples (cum.)  : {len(full_gps_points)}\n")
        f.write(f"GID Medoid Latitude : {medoid_lat}\n")
        f.write(f"GID Medoid Longitude: {medoid_lon}\n\n")

        f.write("------------ This Buffer's Entries ------------\n")

        f.write(
            "Frame      Conf      Sharpness      Latitude      Longitude\n"
        )

        for item in sorted(buffer_gallery, key=lambda x: x[2]):

            f.write(

                f"{item[2]:06d}    "

                f"{item[1]:.4f}    "

                f"{item[6]:8.2f}    "

                f"{item[7]}    "

                f"{item[8]}\n"

            )

        f.write("\n")

    log_ok(
        f"GID={gid} metadata appended ({event}, buffer_gps={len(buffer_gps_points)})."
    )

    log_ok(
        f"GID={gid} Folder ready -> {gid_folder}"
    )


def save_gallery_crops(gid, gallery, tid, frame_id, event):
    global GALLERY_SAVE_DIR   # ← add this line
    """
    Continuous per-buffer-flush dump. One folder per GID; appends new
    crops on both 'new' and 'merge' events. Duplicates skipped by filename.
    Complements save_new_gid_artifacts (which fires once at GID creation).
    """
    if not SAVE_GALLERIES or GALLERY_SAVE_DIR is None:
        return

    folder = os.path.join(GALLERY_SAVE_DIR, f"gid{int(gid)}")
    os.makedirs(folder, exist_ok=True)

    written = skipped = 0
    for item in gallery:
        fid       = int(item[2])
        conf      = float(item[1])
        sharpness = float(item[6])
        crop_img  = item[9] if len(item) > 9 else None

        if crop_img is None or getattr(crop_img, "size", 0) == 0:
            continue

        fname = f"frame{fid:06d}_conf{conf:.2f}_sharp{int(sharpness)}.jpg"
        fpath = os.path.join(folder, fname)
        if os.path.exists(fpath):
            skipped += 1
            continue
        try:
            cv2.imwrite(fpath, crop_img)
            written += 1
        except Exception as e:
            log_warn(f"Gallery crop save failed {fpath}: {e}")

    log_ok(f"[GALLERY DUMP] gid{int(gid)} event={event} "
           f"new={written} skipped={skipped} tid={tid}")


# =========================================================
# GLOBAL ASSOCIATION MODES
# =========================================================

# ---------------------------------------------------------
# USE_GPS = False
# USE_VISUAL_REID = True
# → Pure visual ReID
# ---------------------------------------------------------

# ---------------------------------------------------------
# USE_GPS = True
# USE_VISUAL_REID = False
# → Pure GPS matching
# ---------------------------------------------------------

# ---------------------------------------------------------
# USE_GPS = True
# USE_VISUAL_REID = True
# → Visual + GPS fusion
# ---------------------------------------------------------

USE_GPS         = True
USE_VISUAL_REID = False

# -------- CONFIG VALIDATION --------
if not USE_GPS and not USE_VISUAL_REID:
    raise ValueError(
        "Invalid configuration: "
        "At least one global association method "
        "must be enabled."
    )



# =========================================================
# DETECTION PARAMS (YOLOE — open vocabulary)
# =========================================================

# -------- OPEN-VOCABULARY PROMPTS --------
# YOLOE is conditioned on POSITIVE + NEGATIVE together via set_classes(),
# so the text encoder has both to discriminate against. A detection whose
# predicted class lands in NEGATIVE_CLASSES is NOT a person candidate —
# it's dropped outright, and any person-like box overlapping one is
# dropped too (see suppress_shadow_overlaps() / SHADOW_IOU_SUPPRESS).

POSITIVE_CLASSES = [
    "person",
    # "person",
    # "injured person",
    # "person lying on the ground",
    # "wounded person with visible injuries",
    # "person under tree",
    # "camouflaged person",
]

# Add shadow-type false-positive prompts here — matched by EXACT class
# name (not substring), so wording doesn't matter as long as it's listed.
# Keep this list narrow: broad/ambiguous prompts (e.g. "dark patch on
# ground") compete directly with ambiguous positive prompts like "person
# under tree" / "camouflaged person" in YOLOE's per-box argmax, and start
# swallowing real detections instead of just shadows.
NEGATIVE_CLASSES = [
    "shadow",
    "shadow on ground",
    "dark patch on ground",
    "long shadow",
    "elongated shadow",
    "shadow of a person",
    "cast shadow on ground",
    "dark elongated shape on ground",
]

YOLOE_CLASSES = POSITIVE_CLASSES + NEGATIVE_CLASSES

# -------- INFERENCE SETTINGS --------

YOLOE_CONF   = 0.40   # detection confidence threshold
YOLOE_IOU    = 0.10   # YOLOE internal NMS IoU
YOLOE_IMGSZ  = 640    # inference resolution

YOLOE_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =========================================================
# ROBUST DETECTION (robust_detect / robust_detect_cuda)
# =========================================================
# The legacy path above letterboxes the whole 1920x1080 frame down to
# 640 -- a 0.33x scale -- so a 50 px casualty reaches the network as
# ~17 px and scores accordingly. The robust path runs YOLOE over
# overlapping NATIVE-resolution tiles, fuses the views with weighted
# box fusion, and rejects cast shadows on illumination physics rather
# than on text prompts.
#
# Measured over 20 frames (robust_detect.py), legacy -> robust:
#     mean detection confidence   0.122 -> 0.458
#     detections above 0.50/frame  0.35 -> 2.05
#     boxes emitted per frame       8.4 -> 3.1
#
# Set False to fall back to the legacy single-shot detector. Either
# way the robust path needs .pt weights, because it calls
# set_classes(); with exported weights it disables itself and logs why.


# robust_detect_cuda is the SAME detector with the per-frame path kept
# on the GPU: the unused seg Proto branch skipped, a stride-32
# rectangular letterbox instead of a square one, batched F.interpolate
# preprocessing off a single pinned upload, fp16 + channels_last, and
# the whole forward captured into one CUDA graph. Same tiles, same
# seam rule, same fusion, same photometric constants -- it imports
# them from robust_detect, so there is one copy of the tuning.
# Without CUDA this falls back to the plain module automatically.
ROBUST_USE_CUDA_PATH = True

# "fast"     4 tiles
# "balanced" 9 tiles, the measured quality peak
# "thorough" 9 tiles + a permissive shadow gate
ROBUST_PRESET = "balanced"

# CUDA fast-path knobs. Ignored by the plain module.
ROBUST_PRECISION = "fp16"    # fp16 | bf16 | fp32
ROBUST_BACKEND   = "graph"   # graph | eager | compile
ROBUST_WORKERS   = 8         # threads for the photometric shadow gate

# Where robust_detect.py / robust_detect_cuda.py live. Both the
# pipeline's own directory (deployed layout) and ./testing (repo
# layout) are searched, so this file runs unchanged in either.
_PIPE_DIR = os.path.dirname(os.path.abspath(__file__))
ROBUST_DETECT_DIRS = [_PIPE_DIR, os.path.join(_PIPE_DIR, "testing")]


def _import_robust_detect():
    """
    Returns (module, reference_module, label) or (None, None, reason).

    `module` supplies RobustPersonDetector; `reference_module` is always
    plain robust_detect, which owns the tuned constants both paths share
    (MIN_CONF in particular -- the CUDA module does not re-export it).
    """
    for d in ROBUST_DETECT_DIRS:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)

    try:
        import robust_detect as _rd
    except Exception as e:
        return None, None, f"robust_detect import failed: {e}"

    if ROBUST_USE_CUDA_PATH and torch.cuda.is_available():
        try:
            import robust_detect_cuda as _rdc
            return _rdc, _rd, "cuda"
        except Exception as e:
            print(f"[ROBUST] CUDA fast path unavailable ({e}); "
                  f"using robust_detect")

    return _rd, _rd, "cpu-path"


# Resolved once at import; load_detector() and detect() both read these.
ROBUST_MOD = None      # module providing RobustPersonDetector
ROBUST_REF = None      # plain robust_detect, for shared constants
ROBUST_LABEL = None

if USE_ROBUST_DETECT:

    if not YOLOE_WEIGHTS.endswith(".pt"):
        print(f"[ROBUST] disabled: tiled detection calls set_classes(), "
              f"which exported weights do not expose ({YOLOE_WEIGHTS})")
        USE_ROBUST_DETECT = False

    else:
        ROBUST_MOD, ROBUST_REF, ROBUST_LABEL = _import_robust_detect()

        if ROBUST_MOD is None:
            print(f"[ROBUST] disabled: {ROBUST_LABEL}")
            USE_ROBUST_DETECT = False


def _tune_cuda_backends():
    """
    Backend switches for a fixed-shape vision graph. cudnn.benchmark
    lets cuDNN autotune once per shape and reuse the winner; every
    shape in this pipeline (tiles, 256x128 ReID crops) is constant
    after the first frame, so it is a pure win and changes no result.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

# =========================================================
# MJPEG STREAMING SERVER (DISPLAY_MODE == "stream")
# =========================================================

_output_frame = None     # latest annotated frame, handed to the streamer
_output_seq   = 0        # bumped on every new frame, so the handler can
                         # tell "new frame" from "same frame again"
_frame_lock   = None     # threading.Lock, created by start_stream_server


def start_stream_server(port=STREAM_PORT):
    """
    Serve annotated frames as MJPEG on http://<host>:port/ .

    threading and http.server are imported HERE rather than at module
    scope so a window-mode or headless run neither starts a thread nor
    opens a listening port.
    """
    global _frame_lock

    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn

    _frame_lock = threading.Lock()

    class _MJPEGHandler(BaseHTTPRequestHandler):

        def log_message(self, *args):
            pass                      # keep request spam out of the log

        def do_GET(self):
            self.send_response(200)
            self.send_header(
                "Content-type",
                "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()

            last_seq = -1

            try:
                while True:
                    with _frame_lock:
                        frame, seq = _output_frame, _output_seq

                    # Encode only when there is genuinely a new frame.
                    # The previous version looped on `continue` with no
                    # wait, which pinned a core at 100% before the first
                    # frame arrived and re-encoded the same image
                    # forever after -- CPU taken straight from the
                    # detector this pipeline is trying to keep fed.
                    if frame is None or seq == last_seq:
                        time.sleep(0.005)
                        continue

                    last_seq = seq

                    ok, buf = cv2.imencode(".jpg", frame)
                    if not ok:
                        continue

                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes() + b"\r\n"
                    )

            except (BrokenPipeError, ConnectionResetError):
                pass                  # viewer closed the tab; not an error

    class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        # More than one viewer should not block the others, and the
        # socket must be reusable so a restart is not refused for the
        # TIME_WAIT window.
        daemon_threads = True
        allow_reuse_address = True

    server = _ThreadedHTTPServer(("0.0.0.0", port), _MJPEGHandler)

    threading.Thread(target=server.serve_forever, daemon=True).start()

    return server


REID_BATCH_SIZE = 1

# =========================================================
# IOU THRESHOLDS
# =========================================================

# Track-assignment IoU (detection -> existing track).
IOU_THRESH = 0.25

# =========================================================
# BOUNDING BOX
# =========================================================

BBOX_EXPAND_SCALE = 1.5
CROP_W            = 128
CROP_H            = 256

# =========================================================
# BUFFER LOGIC
# =========================================================

BUFFER_SIZE      = 60
MIN_TRACK_LENGTH = 15

# =========================================================
# REID PARAMETERS
# =========================================================

# -------- GALLERY CONSTRUCTION --------

M = 15   # recent frames
N = 15   # highest confidence
L = 15  # sharpest frames

# -------- FINAL DECISION --------

RATIO_THRESHOLD         = 3.5
MIN_SUPPORT_CONFIDENCE  = 0.78

# -------- FEATURE SELECTION --------

TOP_K_DIMS    = 16
SUPPORT_TOP_K = 3

# -------- VIEWPOINT UNCERTAINTY --------

VIEW_UNCERTAINTY_WEIGHT = 0.35

# -------- SELF BASELINE --------

SELF_SPLIT_REPEATS = 5

# -------- FINAL SIGNAL WEIGHTS --------

W_SEPARATION = 0.35
W_OVERLAP    = 0.20
W_SUPPORT    = 0.45

# -------- NUMERICAL STABILITY --------

EPS = 1e-8


# =========================================================
# GPS / GEO FUSION CONFIG (RTK-OPTIMIZED)
# =========================================================

# -------- FUSION WEIGHTS --------
# Used ONLY when:
# USE_GPS = True
# USE_VISUAL_REID = True

VISUAL_WEIGHT = 0.5
GEO_WEIGHT    = 0.5

# -------- GEO DISTANCE THRESHOLDS (meters) --------

# < GEO_MATCH_THRESH
# → extremely likely same person

# > GEO_FAR_THRESH
# → definitely different

GEO_MATCH_THRESH = 3.0
GEO_FAR_THRESH   = 7.0
# geo_score > 0.5 (the is_match cutoff) sits at
# GEO_MATCH_THRESH + 0.5*(GEO_FAR_THRESH - GEO_MATCH_THRESH) = 5.0m

# Optional hard rejection radius
GEO_RADIUS = GEO_FAR_THRESH

# -------- GALLERY MEDOID (DBSCAN) --------
# robust_gallery_medoid() clusters a gallery's GPS fixes with DBSCAN
# before taking the medoid, so isolated bad fixes (noisy projection,
# momentary misdetection, GPS glitch) never pull the representative
# coordinate off the real cluster of detections.
# Same eps/min_samples as the Stage-1 DBSCAN in
# scp/geolocalization/src/plot_clusters.py, for consistency between
# the two pipelines' clustering behaviour.
GALLERY_DBSCAN_EPS_M       = 1.0   # cluster radius, metres
GALLERY_DBSCAN_MIN_SAMPLES = 3     # min points to form a dense cluster

# =========================================================
# CENTRAL FOV FILTER
# =========================================================

USE_CENTER_FOV_FILTER = True

# Keep only central X% of frame
# 0.5 = central 50%
CENTER_FOV_RATIO = 0.80


# =========================================================
# NORMALISATION
# =========================================================
# Used ONLY for visual ReID branch

NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Normalisation constants kept ON-DEVICE so the ReID batch is converted
# and normalised in two kernels instead of 2N host-side ops per frame.
# Same fp32 arithmetic, same values -- only the place it runs changes.
_NORM_MEAN_DEV = NORM_MEAN.to(device)
_NORM_STD_DEV  = NORM_STD.to(device)

CUDA_OK = device.type == "cuda"

# =========================================================
# LOGGING
# =========================================================

LOG_WIDTH = 72

LOG_FILE = None

def _log(prefix, msg):

    line = f"[{prefix:<7}] {msg}"

    print(line)

    global LOG_FILE

    if LOG_FILE is not None:

        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")

def log_banner(title):
    print("\n" + "=" * LOG_WIDTH)
    print(title.center(LOG_WIDTH))
    print("=" * LOG_WIDTH)

def log_section(title):
    print("\n" + "-" * LOG_WIDTH)
    print(title)
    print("-" * LOG_WIDTH)

def log_ok(msg):
    _log("OK", msg)

def log_info(msg):
    _log("INFO", msg)

def log_warn(msg):
    _log("WARN", msg)

def log_err(msg):
    _log("ERROR", msg)

def log_det(msg):
    _log("DET", msg)

def log_reid(msg):
    _log("REID", msg)

def log_track(msg):
    _log("TRACK", msg)

def log_gps(msg):
    _log("GPS", msg)

def log_match(msg):
    _log("MATCH", msg)

def log_frame(msg):
    _log("FRAME", msg)


# =========================================================
# MODULE 2: MODEL LOADING
# =========================================================
def load_detector():
    """
    YOLOE open-vocabulary detector. ALWAYS required.

    Returns a RobustPersonDetector (tiled + fused + physics-gated) when
    USE_ROBUST_DETECT is set, and the bare YOLOE model otherwise.
    detect() below handles either, so nothing downstream has to care.
    """

    # =========================================================
    # ROBUST (TILED) DETECTOR
    # =========================================================

    if USE_ROBUST_DETECT:

        _tune_cuda_backends()

        # POSITIVE_CLASSES / NEGATIVE_CLASSES are deliberately NOT passed.
        # They belong to the legacy path, where a single "person" prompt
        # plus eight shadow prompts feed the argmax + IoU suppression
        # rule. The robust path is tuned as a unit against its OWN
        # prompt lists and must be run the way the reference pipeline
        # runs it, which is with neither argument supplied:
        #
        #   positives -> robust_detect.POSITIVE_PROMPTS, seven person
        #     wordings. YOLOE takes a max over prompts, so the
        #     pose-specific ones catch prone casualties the generic one
        #     misses; measured at 1280 the seven-prompt set scores
        #     HIGHER than "person" alone, 0.443 vs 0.342.
        #
        #   negatives -> robust_detect.NEGATIVE_PROMPTS, which covers
        #     vegetation and bare soil ("bush", "dark green vegetation",
        #     "patch of bare soil") as well as shadow. These are never
        #     reported; they exist so a dark blob has somewhere better
        #     to land than "person" and takes the pixels away from it in
        #     NMS. The photometric gate does the real rejecting.
        kw = dict(
            weights=YOLOE_WEIGHTS,
            device=YOLOE_DEVICE,
            imgsz=YOLOE_IMGSZ,
        )

        if ROBUST_LABEL == "cuda":
            kw.update(
                precision=ROBUST_PRECISION,
                backend=ROBUST_BACKEND,
                workers=ROBUST_WORKERS,
            )

        det = ROBUST_MOD.make_detector(ROBUST_PRESET, **kw)

        # One warm-up frame at the real resolution: this is what builds
        # the tile plan, allocates the GPU buffers, runs cuDNN's
        # autotuner and captures the CUDA graph, so the first real frame
        # is not paying for any of it. Warm-up counts are then discarded
        # so report() describes the mission, not the warm-up.
        if torch.cuda.is_available():
            det.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
            det.stats = {k: 0 for k in det.stats}
            if hasattr(det, "timing"):
                det.timing = {k: 0.0 for k in det.timing}
            torch.cuda.synchronize()

        return det

    # =========================================================
    # LEGACY SINGLE-SHOT DETECTOR
    # =========================================================

    model = YOLOE(YOLOE_WEIGHTS)

    # .pt carries the text encoder, so the prompts are applied here.
    # Exported weights (.onnx / .engine) have the prompts baked into
    # the graph at export time and expose no text encoder to call,
    # so set_classes() would raise AssertionError on them.

    if YOLOE_WEIGHTS.endswith(".pt"):

        model.set_classes(
            YOLOE_CLASSES,
            model.get_text_pe(YOLOE_CLASSES)
        )

        model.to(device)

    return model

def load_reid():
    """
    Loads the ReID model. Supports:
      - .onnx  → onnxruntime InferenceSession (uses REID_BATCH_SIZE fixed-size batch)
      - .pt / .pth → PyTorch TD_2_NETWORK_V8 checkpoint (dynamic batch)
    Returned object has a uniform .infer(batch_tensor) -> torch.Tensor API
    that prepare_detections_and_embeddings will call.
    """
    if not USE_VISUAL_REID:
        return None

    ext = os.path.splitext(REID_ONNX_PATH)[1].lower()

    # ------------------------------------------------------
    # ONNX BACKEND
    # ------------------------------------------------------
    if ext == ".onnx":
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        sess = ort.InferenceSession(REID_ONNX_PATH, providers=providers)
        log_ok(f"[REID] ONNX backend | providers={sess.get_providers()}")

        class _OnnxReID:
            backend = "onnx"
            def __init__(self, sess):
                self.sess = sess
                # discover input name once
                self.input_name = sess.get_inputs()[0].name

            def infer(self, batch_tensor):
                # batch_tensor: torch.Tensor [N, 3, 256, 128] (CPU or GPU)
                N = batch_tensor.shape[0]
                out_chunks = []
                for start in range(0, N, REID_BATCH_SIZE):
                    end = min(start + REID_BATCH_SIZE, N)
                    chunk = batch_tensor[start:end]
                    chunk_size = chunk.shape[0]

                    batch_np = np.zeros(
                        (REID_BATCH_SIZE, 3, 256, 128),
                        dtype=np.float32,
                    )
                    batch_np[:chunk_size] = chunk.detach().cpu().numpy()

                    raw = self.sess.run(None, {self.input_name: batch_np})[0]
                    out_chunks.append(raw[:chunk_size])

                raw_all = np.concatenate(out_chunks, axis=0)
                return torch.from_numpy(raw_all).float().to(device)

        return _OnnxReID(sess)

    # ------------------------------------------------------
    # PYTORCH BACKEND
    # ------------------------------------------------------
    elif ext in (".pt", ".pth"):
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(REID_ONNX_PATH)))
        try:
            from TD_Resnet_v3 import TD_2_NETWORK_V8
        except ImportError as e:
            raise RuntimeError(
                f"PyTorch ReID backend needs TD_Resnet_v3.py alongside "
                f"the checkpoint. Import failed: {e}"
            )

        model = TD_2_NETWORK_V8(num_classes=1580).to(device)
        ckpt = torch.load(REID_ONNX_PATH, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        # strip DataParallel 'module.' prefix if present
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model.eval()

        _tune_cuda_backends()

        # One warm-up pass at the exact crop shape, so the first real
        # frame is not paying for cuDNN algorithm selection and kernel
        # autotuning. Every ReID batch is [N, 3, 256, 128] with only N
        # varying, so the autotuned plans are reused from here on.
        if device.type == "cuda":
            with torch.inference_mode():
                model(torch.zeros(1, 3, CROP_H, CROP_W, device=device))
            torch.cuda.synchronize()

        log_ok(f"[REID] PyTorch backend | device={device}")

        class _TorchReID:
            backend = "torch"
            def __init__(self, model):
                self.model = model

            @torch.inference_mode()
            def infer(self, batch_tensor):
                # batch_tensor: torch.Tensor [N, 3, 256, 128], already
                # normalised and already on `device` on the CUDA path,
                # so the .to() below is a no-op there rather than a
                # second copy. inference_mode beats no_grad: it also
                # skips version-counter bookkeeping on every tensor.
                batch = batch_tensor.to(device, non_blocking=True)
                return self.model(batch).float()

        return _TorchReID(model)

    else:
        raise ValueError(
            f"Unsupported ReID checkpoint extension: {ext!r} "
            f"(expected .onnx, .pt, or .pth)"
        )

# =========================================================
# MODULE 3: GLOBAL TRACKING STRUCTURES
# =========================================================

# tracks[track_id] = [[emb, conf, frame_id, cx, cy, s, sharpness, lat, lon], ...]

# REID_DICT[global_id] = [[emb, conf, frame_id, cx, cy, s, sharpness, lat, lon], ...]

# moving_tracks[track_id] = [[emb, conf, frame_id, cx, cy, s, sharpness, lat, lon, global_id, track_id], ...]

# emb     = None when USE_VISUAL_REID == False
# lat/lon = None when USE_GPS == False

tracks        = {}
REID_DICT     = {}
moving_tracks = {}

# GID_GPS_HISTORY[gid] = [(lat, lon), ...] -- every GPS fix from every
# buffer ever assigned to this GID (new + all merges), even ones later
# pruned out of REID_DICT[gid]["gallery"] by the M/N/L cap. This is the
# full-history point cloud save_new_gid_artifacts DBSCANs to produce the
# GID's "Final GPS Coordinate" in metadata.txt.
GID_GPS_HISTORY = {}

### DEBUGGING GPS

GPS_DEBUG = {}

# -------- FRAME STORAGE --------
# FRAME_BUFFER[frame_id] = frame

FRAME_BUFFER = {}

# -------- REFERENCE COUNTING --------
# ACTIVE_FRAMES[frame_id] = number of tracks using this frame

ACTIVE_FRAMES = {}

next_track_id  = 1
next_global_id = 1



# =========================================================
# MODULE 4: CORE UTILITY FUNCTIONS
# =========================================================

def detect(detector, frame):
    """
    Returns [(x1, y1, x2, y2, conf, cls_name), ...] in original frame coords.

    Accepts either detector kind -- the tiled RobustPersonDetector or the
    legacy bare YOLOE model -- and returns the same 6-tuples, so
    suppress_shadow_overlaps() and everything after it stay unchanged.

    In robust mode cls_name is the positive prompt the strongest view
    fired on. Negative-class boxes never reach here: the robust path
    consumes them internally as NMS decoys and rejects shadows on the
    photometric gate instead, which is a strictly stronger test than the
    IoU overlap rule (held-out AUC 0.907 vs YOLOE's own 0.606). That
    leaves suppress_shadow_overlaps() a harmless pass-through rather
    than a second, weaker opinion.
    """

    # =========================================================
    # ROBUST (TILED) PATH
    # =========================================================

    if USE_ROBUST_DETECT and hasattr(detector, "detect"):

        # The plain module applies robust_detect.MIN_CONF as a final
        # gate on the score that actually leaves detect(); the CUDA
        # module does not implement it. Applying it here is identical
        # either way -- it is the same number on the same post-physics
        # confidence -- and keeps the two paths reporting the same
        # boxes. This is NOT CONF_FLOOR: that one cuts raw per-tile
        # scores before fusion, and raising it would throw away exactly
        # the weak-but-corroborated boxes tiling exists to rescue.
        min_conf = getattr(ROBUST_REF, "MIN_CONF", 0.0)

        return [
            (d.x1, d.y1, d.x2, d.y2, d.conf, d.prompt or "person")
            for d in detector.detect(frame)
            if d.conf >= min_conf
        ]

    # =========================================================
    # LEGACY SINGLE-SHOT PATH
    # =========================================================

    result = detector.predict(
        frame,
        imgsz=YOLOE_IMGSZ,
        conf=YOLOE_CONF,
        iou=YOLOE_IOU,
        device=YOLOE_DEVICE,
        verbose=False
    )[0]

    names = result.names   # {id: class_name}
    boxes = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id   = int(box.cls[0])
        cls_name = names.get(cls_id, "unknown")
        boxes.append((
            int(x1), int(y1), int(x2), int(y2),
            float(box.conf[0]),
            cls_name,
        ))

    return boxes


def make_square_bbox(x1, y1, x2, y2, img_w, img_h):
    w = x2 - x1
    h = y2 - y1

    if w > h:
        diff = w - h
        y1 -= diff // 2
        y2 += diff - diff // 2

    else:
        diff = h - w
        x1 -= diff // 2
        x2 += diff - diff // 2

    return (
        int(max(0, x1)),
        int(max(0, y1)),
        int(min(img_w, x2)),
        int(min(img_h, y2))
    )


def expand_bbox(x1, y1, x2, y2, img_w, img_h):
    w     = x2 - x1
    cx    = (x1 + x2) // 2
    cy    = (y1 + y2) // 2
    new_w = int(w * BBOX_EXPAND_SCALE)

    x1 = cx - new_w // 2
    y1 = cy - new_w // 2
    x2 = cx + new_w // 2
    y2 = cy + new_w // 2

    return (
        int(max(0, x1)),
        int(max(0, y1)),
        int(min(img_w, x2)),
        int(min(img_h, y2))
    )


# =========================================================
# SHARPNESS
# =========================================================

def compute_sharpness(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    return cv2.Laplacian(gray, cv2.CV_64F).var()


# =========================================================
# GEO FUNCTIONS
# =========================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Returns distance in meters between two GPS points.
    """

    R = 6371000  # Earth radius in meters

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def geo_score_from_distance(dist):
    """
    RTK-optimized geo scoring.

    < GEO_MATCH_THRESH
        → definite match

    > GEO_FAR_THRESH
        → definite reject

    otherwise
        → linear decay
    """

    if dist < GEO_MATCH_THRESH:
        return 1.0

    elif dist > GEO_FAR_THRESH:
        return 0.0

    else:
        return 1.0 - (
            (dist - GEO_MATCH_THRESH)
            / (GEO_FAR_THRESH - GEO_MATCH_THRESH)
        )



def robust_gallery_medoid(gps_points):
    """
    Computes a robust representative GPS coordinate for a gallery.

    Was: medoid, then reject anything > 10m from it, then re-medoid --
    a fixed-radius heuristic that has no notion of point density.

    Now: DBSCAN-cluster the gallery's GPS fixes in local metres, drop
    everything DBSCAN calls noise (isolated bad fixes from a noisy
    projection, a momentary misdetection, a GPS glitch, etc.), keep only
    the largest dense cluster as "the person", and return the haversine
    medoid of just that cluster. Falls back to a plain medoid over every
    point if DBSCAN can't form any cluster (too few / too scattered
    points) so this never returns None while points exist.

    Returns:
        representative_lat,
        representative_lon
    """

    if len(gps_points) == 0:
        return None, None

    if len(gps_points) == 1:
        return gps_points[0]

    # =====================================================
    # HELPER: FIND MEDOID (haversine-based)
    # =====================================================

    def compute_medoid(points):

        best_point = None
        best_score = float("inf")

        for i, (lat1, lon1) in enumerate(points):

            total_dist = 0.0

            for j, (lat2, lon2) in enumerate(points):

                if i == j:
                    continue

                total_dist += haversine_distance(
                    lat1,
                    lon1,
                    lat2,
                    lon2
                )

            if total_dist < best_score:

                best_score = total_dist
                best_point = (lat1, lon1)

        return best_point

    # =====================================================
    # LAT/LON -> LOCAL METRES (equirectangular approx --
    # accurate enough at gallery scale, i.e. a few metres)
    # =====================================================

    lats = np.array([p[0] for p in gps_points])
    lons = np.array([p[1] for p in gps_points])

    lat0 = lats.mean()
    lon0 = lons.mean()

    EARTH_R = 6_371_000.0

    x = np.radians(lons - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lats - lat0) * EARTH_R

    xy = np.column_stack([x, y])

    # =====================================================
    # DBSCAN: drop noisy/outlier fixes, keep only the
    # densest cluster of detections as the true location
    # =====================================================

    labels = DBSCAN(
        eps=GALLERY_DBSCAN_EPS_M,
        min_samples=GALLERY_DBSCAN_MIN_SAMPLES,
    ).fit_predict(xy)

    cluster_ids = [l for l in set(labels) if l != -1]

    if not cluster_ids:
        # Nothing dense enough (too few/scattered points) -- fall back
        # to the raw medoid over every point instead of failing.
        return compute_medoid(gps_points)

    largest_id = max(cluster_ids, key=lambda l: int((labels == l).sum()))

    kept_points = [
        gps_points[i]
        for i in range(len(gps_points))
        if labels[i] == largest_id
    ]

    return compute_medoid(kept_points)



# =========================================================
# IOU
# =========================================================

def compute_iou_matrix(boxes_tensor):
    x = boxes_tensor[:, 0].unsqueeze(1)
    y = boxes_tensor[:, 1].unsqueeze(1)
    s = boxes_tensor[:, 2].unsqueeze(1)

    del_x = torch.abs(x - x.T)
    del_y = torch.abs(y - y.T)

    side = (s + s.T) / 2

    inter = (
        torch.clamp(side - del_x, min=0)
        * torch.clamp(side - del_y, min=0)
    )

    union = s * s + s.T * s.T - inter

    return inter / (union + 1e-6)


# =========================================================
# MODULE 5: CROP PREPROCESSING
# =========================================================
#
# Mirrors test script's transform pipeline exactly:
#
# transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))
#     → cv2.resize(crop, (CROP_W, CROP_H))
#
# transforms.ToTensor()
#     → permute + /255.0
#
# transforms.Normalize(mean, std)
#     → (crop - mean) / std
#
# NOTE:
# cv2.resize takes (width, height)
#
# so:
# (CROP_W, CROP_H) = (128, 256)
#
# producing:
# [3, 256, 128]
#
# Used ONLY when:
# USE_VISUAL_REID == True
#
# =========================================================

def preprocess_crop(crop_bgr):
    """
    Converts a BGR numpy crop into a uint8 CHW tensor:
        [3, CROP_H, CROP_W]

    The /255 and the mean/std normalisation are deliberately NOT done
    here. They are applied once to the whole stacked batch (see the
    embedding extraction below), which is the same fp32 arithmetic on
    the same values but two kernels per frame instead of two host-side
    ops per detection -- and it keeps the host->device copy at one byte
    per channel instead of four.

    torch.from_numpy shares the buffer with numpy rather than copying
    element by element the way torch.tensor() on a list-like does.
    """

    crop = cv2.resize(crop_bgr, (CROP_W, CROP_H))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    return torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)


# =========================================================
# MODULE 6: DETECTION FILTERING + FEATURE EXTRACTION
# =========================================================

# =========================================================
# SHADOW SUPPRESSION
# =========================================================

# NEGATIVE_CLASSES (config section above) is the source of truth for
# which YOLOE prompts count as "shadow-type" — exact class-name match.
SHADOW_IOU_SUPPRESS = 0.3   # overlap ratio above which a person box is dropped


def _box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def suppress_shadow_overlaps(boxes):
    """
    boxes: list of (x1, y1, x2, y2, conf, cls_name)
    Returns: filtered list with negative-class (shadow) boxes removed AND
             any person-like box that overlaps one above SHADOW_IOU_SUPPRESS.
    """
    shadow_boxes = [b for b in boxes if b[5] in NEGATIVE_CLASSES]
    person_boxes = [b for b in boxes if b[5] not in NEGATIVE_CLASSES]

    if not shadow_boxes:
        return person_boxes

    kept = []
    dropped = 0
    for pb in person_boxes:
        overlap = max(
            (_box_iou_xyxy(pb[:4], sb[:4]) for sb in shadow_boxes),
            default=0.0
        )
        if overlap >= SHADOW_IOU_SUPPRESS:
            dropped += 1
            continue
        kept.append(pb)

    if dropped:
        log_det(f"[SHADOW SUPPRESS] dropped {dropped} person-like boxes "
                f"overlapping {len(shadow_boxes)} shadow boxes")

    return kept


def prepare_detections_and_embeddings(frame, boxes, reid_model, frame_id, frame_timestamp, telem):
    h, w = frame.shape[:2]

    processed_boxes = []
    crops           = []

    for (x1, y1, x2, y2, conf) in boxes:

        # =========================================================
        # CENTRAL FOV FILTER
        # =========================================================

        if USE_CENTER_FOV_FILTER:

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            valid_w = w * CENTER_FOV_RATIO
            valid_h = h * CENTER_FOV_RATIO

            valid_x1 = (w - valid_w) / 2
            valid_y1 = (h - valid_h) / 2

            valid_x2 = valid_x1 + valid_w
            valid_y2 = valid_y1 + valid_h

            # Reject detections whose CENTER
            # lies outside central valid region

            if (
                cx < valid_x1 or
                cx > valid_x2 or
                cy < valid_y1 or
                cy > valid_y2
            ):
                MISSION_STATS["fov_rejects"] += 1
                continue

        # Raw detector box, kept for geolocalisation: squaring/expanding
        # clips at the frame edge, which shifts the box centre.
        det_x1, det_y1, det_x2, det_y2 = x1, y1, x2, y2

        x1, y1, x2, y2 = make_square_bbox(x1, y1, x2, y2, w, h)
        x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h)

        # -------- DISCARD INVALID BOXES --------
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            MISSION_STATS["invalid_rejects"] += 1
            continue

        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            MISSION_STATS["invalid_rejects"] += 1
            continue

        crop = frame[y1:y2, x1:x2]
        crop_copy = crop.copy()

        if crop.size == 0:
            MISSION_STATS["invalid_rejects"] += 1
            continue

        # =========================================================
        # SHARPNESS
        # =========================================================

        sharpness = compute_sharpness(crop)

        if sharpness < 20:
            MISSION_STATS["blur_rejects"] += 1
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        brightness = gray.mean()
        contrast   = gray.std()

        # =========================================================
        # SHADOW FILTERING
        # =========================================================
        
        if brightness < 30:
            MISSION_STATS["shadow_rejects"] += 1
            continue

        if contrast < 10:
            MISSION_STATS["shadow_rejects"] += 1
            continue

        # =========================================================
        # GPS
        # =========================================================

        if USE_GPS:

            try:

                lat, lon = bbox_to_gps(
                    det_x1,
                    det_y1,
                    det_x2,
                    det_y2,
                    telem
                )

            except Exception as e:

                log_err(f"GPS ERROR: {e}")  
                MISSION_STATS["gps_failures"] += 1

                lat, lon = None, None

        else:
            lat, lon = None, None

        # =========================================================
        # VISUAL REID PREPROCESSING
        # =========================================================

        if USE_VISUAL_REID:
            crops.append(preprocess_crop(crop))


        # =========================================================
        # ALTITUDE FILTER
        # =========================================================

      

        # =========================================================
        # STORE
        # =========================================================

        processed_boxes.append(
            (x1, y1, x2, y2, conf, sharpness, lat, lon, crop_copy)
        )

    # =========================================================
    # GPS-ONLY MODE
    # =========================================================

    if not USE_VISUAL_REID:
        return processed_boxes, [None] * len(processed_boxes)

    # =========================================================
    # NO VALID CROPS
    # =========================================================

    if not crops:
        
        return [], torch.empty((0, 512), device=device)


    # =========================================================
    # EMBEDDING EXTRACTION
    # =========================================================

    # =========================================================
    # EMBEDDING EXTRACTION (backend-agnostic)
    # =========================================================

    batch = torch.stack(crops)   # [N, 3, 256, 128] uint8, CPU

    if reid_model.backend == "torch" and CUDA_OK:

        # One pinned, non-blocking host->device copy for the whole
        # batch, then convert and normalise on the GPU. The ONNX
        # backend wants its input back on the host anyway, so it keeps
        # doing this on the CPU -- moving it to the GPU there would only
        # buy a round trip.
        batch = batch.pin_memory().to(device, non_blocking=True)
        batch = batch.float().div_(255.0)
        batch = (batch - _NORM_MEAN_DEV) / _NORM_STD_DEV

    else:

        batch = batch.float().div_(255.0)
        batch = (batch - NORM_MEAN) / NORM_STD

    raw_embeddings = reid_model.infer(batch)  # -> torch.Tensor on `device`

    embeddings = F.normalize(raw_embeddings, dim=1)

    return processed_boxes, embeddings


# =========================================================
# MODULE 7: IOU-BASED TRACK ASSIGNMENT
# =========================================================

def assign_detections_to_tracks(valid_boxes, embeddings, frame_id):
    global tracks, next_track_id, ACTIVE_FRAMES

    if not valid_boxes:
        return

    # -------- IGNORE EXTRA METADATA --------
    det_centers = torch.tensor(
        [[(x1+x2)/2, (y1+y2)/2, x2-x1] for x1, y1, x2, y2, *_ in valid_boxes],
        dtype=torch.float32,
        device=device
    )

    track_ids = list(tracks.keys())

    if track_ids:

        track_centers = torch.tensor(
            [[t[-1][3], t[-1][4], t[-1][5]] for t in tracks.values()],
            dtype=torch.float32,
            device=device
        )

        all_boxes  = torch.cat([det_centers, track_centers], dim=0)
        iou_full   = compute_iou_matrix(all_boxes)
        iou_matrix = iou_full[:len(valid_boxes), len(valid_boxes):]

    else:
        iou_matrix = torch.zeros((len(valid_boxes), 0), device=device)

    det_to_track   = [-1] * len(valid_boxes)
    track_assigned = set()

    # =========================================================
    # IOU ASSOCIATION
    # =========================================================

    for d in range(len(valid_boxes)):

        if iou_matrix.shape[1] == 0:
            break

        ious               = iou_matrix[d]
        best_iou, best_idx = torch.max(ious, dim=0)

        if best_iou.item() < IOU_THRESH:
            continue

        close = torch.where(ious > best_iou * 0.9)[0]

        if len(close) > 1:
            det_to_track[d] = -2
            continue

        if best_idx.item() in track_assigned:
            det_to_track[d] = -2

        else:
            det_to_track[d] = best_idx.item()
            track_assigned.add(best_idx.item())

    # =========================================================
    # APPLY ASSIGNMENTS
    # =========================================================

    for i, assignment in enumerate(det_to_track):

        emb = embeddings[i]

        x1, y1, x2, y2, conf, sharpness, lat, lon, crop_img = valid_boxes[i]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s  = x2 - x1

        # -------- FRAME REFERENCE COUNT --------
        if frame_id not in ACTIVE_FRAMES:
            ACTIVE_FRAMES[frame_id] = 0

        ACTIVE_FRAMES[frame_id] += 1

        # =========================================================
        # NEW TRACK
        # =========================================================

        if assignment == -1:

            tracks[next_track_id] = [
                [emb, conf, frame_id, cx, cy, s,
                 sharpness, lat, lon, crop_img]
            ]

            MISSION_STATS["tracks_created"] += 1
            next_track_id += 1

        # =========================================================
        # EXISTING TRACK
        # =========================================================

        elif assignment >= 0:

            tid = track_ids[assignment]

            # =====================================================
            # NORMAL APPEND
            # =====================================================

            tracks[tid].append(
                [emb, conf, frame_id, cx, cy, s,
                sharpness, lat, lon, crop_img]
            )
            MISSION_STATS["tracks_updated"] += 1

        # =========================================================
        # -2 → ambiguous assignment → discard
        # =========================================================
                

        
# =========================================================
# MODULE 8: BUFFER MANAGEMENT + GALLERY EXTRACTION
# =========================================================

def process_track_buffers(frame_id):
    global tracks

    reid_candidates  = []
    tracks_to_delete = []

    for tid, history in tracks.items():

        birth_frame = history[0][2]

        # =========================================================
        # BUFFER NOT READY
        # =========================================================

        if (frame_id - birth_frame + 1) < BUFFER_SIZE:
            continue

        # =========================================================
        # TOO SHORT → DISCARD
        # =========================================================

        if len(history) < MIN_TRACK_LENGTH:
            tracks_to_delete.append(tid)
            continue

        # =========================================================
        # RECENT (M)
        # =========================================================

        recent = sorted(
            history[-M:],
            key=lambda x: x[2]
        )

        # =========================================================
        # TOP CONFIDENCE (N)
        # =========================================================

        top_conf = sorted(
            sorted(
                history,
                key=lambda x: x[1],
                reverse=True
            )[:N],
            key=lambda x: x[2]
        )

        # =========================================================
        # TOP SHARPNESS (L)
        # =========================================================
        # x[6] = sharpness

        top_sharp = sorted(
            sorted(
                history,
                key=lambda x: x[6],
                reverse=True
            )[:L],
            key=lambda x: x[2]
        )

        # =========================================================
        # FINAL GALLERY
        # =========================================================

        gallery = recent + top_conf + top_sharp

        # =========================================================
        # REMOVE DUPLICATES
        # =========================================================
        # IMPORTANT:
        # weighted cosine support becomes biased
        # if same frame appears multiple times

        unique_gallery = []
        seen_frames    = set()

        for item in gallery:

            fid = item[2]

            if fid in seen_frames:
                continue

            seen_frames.add(fid)

            unique_gallery.append(item)

        # =========================================================
        # SORT FINAL GALLERY TEMPORALLY
        # =========================================================

        gallery = sorted(
            unique_gallery,
            key=lambda x: x[2]
        )

        # =========================================================
        # FINAL SAFETY CHECK
        # =========================================================

        if len(gallery) < 2:
            tracks_to_delete.append(tid)
            continue

        reid_candidates.append((tid, gallery))

        # =========================================================
        # RESET TRACK
        # =========================================================
        # keep only latest frame
        # preserves continuity without unbounded growth

        tracks[tid] = [history[-1]]

    # =========================================================
    # DELETE SHORT TRACKS
    # =========================================================

    for tid in tracks_to_delete:

        if tid in tracks:
            del tracks[tid]
            MISSION_STATS["tracks_deleted"] += 1

    return reid_candidates




# =========================================================
# MODULE 9: GLOBAL ASSOCIATION COMPARISON
# =========================================================

# =========================================================
# RAW WEIGHTED SEPARATION SCORE
# =========================================================

def raw_separation_score(gallery_a, gallery_b):

    if not USE_VISUAL_REID:
        return 0.0, 0.0

    if len(gallery_a) == 0 or len(gallery_b) == 0:
        return 0.0, 0.0

    # =====================================================
    # EMBEDDINGS
    # =====================================================

    E1_np = torch.stack([item[0] for item in gallery_a]).cpu().numpy()
    E2_np = torch.stack([item[0] for item in gallery_b]).cpu().numpy()

    # =====================================================
    # DIFFERENCE DISTRIBUTION
    # =====================================================

    diffs = (
        E1_np[:, np.newaxis, :]
        - E2_np[np.newaxis, :, :]
    ).reshape(-1, E1_np.shape[1])

    # =====================================================
    # SNR
    # =====================================================

    mu = np.mean(diffs, axis=0)

    sigma = np.std(diffs, axis=0) + EPS

    snr = np.abs(mu) / sigma

    # =====================================================
    # MUTUAL SUPPORT
    # =====================================================

    support_strength = np.mean(
        np.abs(
            E1_np[:, np.newaxis, :]
            * E2_np[np.newaxis, :, :]
        ),
        axis=(0, 1)
    )

    # =====================================================
    # NORMALIZE
    # =====================================================

    snr_norm = snr / (np.max(snr) + EPS)

    support_norm = (
        support_strength
        / (np.max(support_strength) + EPS)
    )

    # =====================================================
    # COMBINED DIMENSION SCORE
    # =====================================================

    dim_score = (
        0.45 * snr_norm
        + 0.55 * support_norm
    )

    top_dims = np.argsort(dim_score)[::-1][:TOP_K_DIMS]

    # =====================================================
    # SIGNAL 1 — SEPARATION
    # =====================================================

    s_sep = float(
        np.mean(
            np.tanh(snr[top_dims])
        )
    )

    # =====================================================
    # SIGNAL 2 — OVERLAP
    # =====================================================

    m1 = np.mean(E1_np, axis=0)
    m2 = np.mean(E2_np, axis=0)

    s1 = np.std(E1_np, axis=0)
    s2 = np.std(E2_np, axis=0)

    z = np.abs(m1 - m2) / (s1 + s2 + EPS)

    separation = 1.0 - np.exp(-0.5 * z**2)

    s_overlap = float(
        np.mean(separation[top_dims])
    )

    # =====================================================
    # SIGNAL 3 — SUPPORT CONSISTENCY
    # =====================================================

    A = E1_np[:, top_dims]
    B = E2_np[:, top_dims]

    A = A / (
        np.linalg.norm(A, axis=1, keepdims=True)
        + EPS
    )

    B = B / (
        np.linalg.norm(B, axis=1, keepdims=True)
        + EPS
    )

    sim = A @ B.T

    topA_vals = np.sort(sim, axis=1)[:, -SUPPORT_TOP_K:]
    topB_vals = np.sort(sim.T, axis=1)[:, -SUPPORT_TOP_K:]

    topA = np.mean(topA_vals ** 2)
    topB = np.mean(topB_vals ** 2)

    support = (topA + topB) / 2.0

    s_support = 1.0 - support

    # =====================================================
    # FINAL SCORE
    # =====================================================

    final_score = (
        W_SEPARATION * s_sep
        + W_OVERLAP * s_overlap
        + W_SUPPORT * s_support
    )

    return float(final_score), float(support)


# =========================================================
# SELF BASELINE
# =========================================================

def self_baseline_score(gallery):

    if not USE_VISUAL_REID:
        return 0.0

    if len(gallery) < 2:
        return 0.0

    scores = []

    rng = np.random.default_rng(42)

    n = len(gallery)

    for _ in range(SELF_SPLIT_REPEATS):

        idx = rng.permutation(n)

        half = max(1, n // 2)

        g1 = [gallery[i] for i in idx[:half]]

        g2 = [gallery[i] for i in idx[half:]]

        if len(g2) == 0:
            g2 = g1

        score, _ = raw_separation_score(g1, g2)

        scores.append(score)

    return float(np.mean(scores))


# =========================================================
# MAIN GALLERY COMPARISON
# =========================================================

def compare_galleries(gallery_a, gallery_b):
    """
    Supports:

    1. Visual-only
    2. GPS-only
    3. Visual + GPS fusion
    """

    # =========================================================
    # GPS-ONLY MODE
    # =========================================================

    if USE_GPS and not USE_VISUAL_REID:

        gps_a = [
            (item[7], item[8])
            for item in gallery_a
            if item[7] is not None and item[8] is not None
        ]

        gps_b = [
            (item[7], item[8])
            for item in gallery_b
            if item[7] is not None and item[8] is not None
        ]

        if len(gps_a) == 0 or len(gps_b) == 0:
            return 0.0, False

        robust_lat_a, robust_lon_a = robust_gallery_medoid(gps_a)

        robust_lat_b, robust_lon_b = robust_gallery_medoid(gps_b)

        dist = haversine_distance(
            robust_lat_a,
            robust_lon_a,
            robust_lat_b,
            robust_lon_b
        )

        geo_score = geo_score_from_distance(dist)

        return geo_score, geo_score > 0.5

    # =========================================================
    # WEIGHTED VISUAL COMPARISON
    # =========================================================

    cross_score, support = raw_separation_score(
        gallery_a,
        gallery_b
    )

    baseline_a = self_baseline_score(gallery_a)
    baseline_b = self_baseline_score(gallery_b)

    # =========================================================
    # VIEWPOINT UNCERTAINTY
    # =========================================================

    E1 = torch.stack([item[0] for item in gallery_a]).to(device)
    E2 = torch.stack([item[0] for item in gallery_b]).to(device)

    c1 = E1.mean(dim=0)
    c2 = E2.mean(dim=0)

    c1 = F.normalize(c1.unsqueeze(0), dim=1).squeeze(0)
    c2 = F.normalize(c2.unsqueeze(0), dim=1).squeeze(0)

    centroid_sim = torch.dot(c1, c2).item()

    view_uncertainty = 1.0 - centroid_sim

    # =========================================================
    # SUPPORT GATE
    # =========================================================

    support_gate = np.clip(
        (support - 0.55) / 0.25,
        0.0,
        1.0
    )

    # =========================================================
    # EFFECTIVE BASELINE
    # =========================================================

    effective_baseline = (
        max(baseline_a, baseline_b)
        + VIEW_UNCERTAINTY_WEIGHT
        * support_gate
        * view_uncertainty
    )

    # =========================================================
    # RATIO
    # =========================================================

    ratio = cross_score / (
        effective_baseline + EPS
    )

    # =========================================================
    # GPS FUSION
    # =========================================================

    if USE_GPS:

        gps_a = [
            (item[7], item[8])
            for item in gallery_a
            if item[7] is not None and item[8] is not None
        ]

        gps_b = [
            (item[7], item[8])
            for item in gallery_b
            if item[7] is not None and item[8] is not None
        ]

        if len(gps_a) > 0 and len(gps_b) > 0:

            lat1, lon1 = robust_gallery_medoid(gps_a)
            lat2, lon2 = robust_gallery_medoid(gps_b)

            dist = haversine_distance(
                lat1,
                lon1,
                lat2,
                lon2
            )

            if GEO_WEIGHT > 0 and dist > GEO_FAR_THRESH:
                return ratio, False

            geo_score = geo_score_from_distance(dist)

        else:
            geo_score = 0.0

    else:
        geo_score = 0.0

    # =========================================================
    # FINAL VISUAL SCORE
    # =========================================================

    visual_match = (
        ratio < RATIO_THRESHOLD
        and
        support > MIN_SUPPORT_CONFIDENCE
    )

    # =========================================================
    # FINAL DECISION
    # =========================================================

    if not USE_GPS:

        is_match = visual_match

    else:

        visual_score = np.clip(
            1.0 - (ratio / RATIO_THRESHOLD),
            0.0,
            1.0
        )

        final_score = (
            VISUAL_WEIGHT * visual_score
            + GEO_WEIGHT * geo_score
        )

        is_match = (
            final_score > 0.5
            and visual_match
        )

    # =========================================================
    # DEBUG
    # =========================================================

    log_match(
        f"[WEIGHTED MATCH] "
        f"cross={cross_score:.4f} "
        f"ratio={ratio:.3f} "
        f"support={support:.3f} "
        f"baselineA={baseline_a:.4f} "
        f"baselineB={baseline_b:.4f} "
        f"match={is_match}"
    )

    return ratio, is_match




# =========================================================
# MODULE 10: GLOBAL ID ASSIGNMENT
# =========================================================

def process_reid_candidates(reid_candidates, ros_time, frame_id):
    global REID_DICT, moving_tracks
    global next_global_id, tracks

    # =========================================================
    # REMOVE STALE moving_tracks
    # =========================================================

    valid_tids = set(tracks.keys())

    for mt_tid in [k for k in list(moving_tracks) if k not in valid_tids]:
        del moving_tracks[mt_tid]

    # =========================================================
    # ACTIVE GLOBAL IDS
    # =========================================================

    active_gids = {
        int(mt_hist[-1][10])
        for mt_hist in moving_tracks.values()
    }

    # =========================================================
    # PASS 1: COLLECT CANDIDATE ASSIGNMENTS
    # =========================================================

    temp_assignments = {}
    temp_new_flags   = {}
    temp_galleries   = {}

    for (tid, gallery) in reid_candidates:

        temp_galleries[tid] = gallery

        # =========================================================
        # REUSE EXISTING ACTIVE GID
        # =========================================================

        matched_gid = None

        for mt_hist in moving_tracks.values():

            mt_last = mt_hist[-1]

            if mt_last[11] == tid:
                matched_gid = int(mt_last[10])
                break

        if matched_gid is not None:

            temp_assignments[tid] = matched_gid
            temp_new_flags[tid]   = False

            continue

        # =========================================================
        # COMPARE AGAINST STORED GLOBAL IDS
        # =========================================================

        matched = []

        for gid, stored_data in REID_DICT.items():

            gid = int(gid)

            stored_gallery  = stored_data["gallery"]
            stored_baseline = stored_data["baseline"]

            # =====================================================
            # SKIP ACTIVE GIDS
            # =====================================================

            if gid in active_gids:
                continue

            # =====================================================
            # GPS PRE-FILTER
            # =====================================================

            if USE_GPS and GEO_WEIGHT > 0:

                gps_a = [
                    (item[7], item[8])
                    for item in gallery
                    if item[7] is not None and item[8] is not None
                ]

                gps_b = [
                    (item[7], item[8])
                    for item in stored_gallery
                    if item[7] is not None and item[8] is not None
                ]

                # -------------------------------------------------
                # skip if no valid GPS
                # -------------------------------------------------

                if len(gps_a) == 0 or len(gps_b) == 0:
                    continue

                # -------------------------------------------------
                # ROBUST MEDIAN GPS
                # -------------------------------------------------

                lat1, lon1 = robust_gallery_medoid(gps_a)

                lat2, lon2 = robust_gallery_medoid(gps_b)

                dist = haversine_distance(
                    lat1,
                    lon1,
                    lat2,
                    lon2
                )

                log_gps(
                    f"[GPS PREFILTER] "
                    f"TID={tid} "
                    f"vs GID={gid} | "
                    f"TRACK_MEDOID=({lat1:.6f}, {lon1:.6f}) "
                    f"STORED_MEDOID=({lat2:.6f}, {lon2:.6f}) "
                    f"DIST={dist:.2f}m"
                )

                if dist > GEO_FAR_THRESH:

                    log_gps(
                        f"[GPS REJECTED] "
                        f"TID={tid} "
                        f"vs GID={gid} | "
                        f"DIST={dist:.2f}m > "
                        f"{GEO_FAR_THRESH}m"
                    )

                    continue

            # =====================================================
            # WEIGHTED GALLERY COMPARISON
            # =====================================================

            if USE_VISUAL_REID:
                candidate_baseline = self_baseline_score(gallery)

                ratio, is_match = compare_galleries(
                    gallery,
                    stored_gallery
                )

                log_match(
                    f"[GLOBAL MATCH] "
                    f"TID={tid} "
                    f"vs GID={gid} | "
                    f"Na={len(gallery)} "
                    f"Nb={len(stored_gallery)} "
                    f"pairs={len(gallery)*len(stored_gallery)} "
                    f"baselineA={candidate_baseline:.4f} "
                    f"baselineB={stored_baseline:.4f} "
                    f"ratio={ratio:.5f} "
                    f"match={is_match}"
                )
            else:
                ratio, is_match = compare_galleries(
                    gallery,
                    stored_gallery
                )

                log_match(
                    f"[GPS MATCH] "
                    f"TID={tid} "
                    f"vs GID={gid} | "
                    f"geo_score={ratio:.4f} "
                    f"match={is_match}"
                )

            if is_match:
                matched.append(gid)

        # =========================================================
        # ASSIGNMENT LOGIC
        # =========================================================

        if len(matched) == 0:


            gid = int(next_global_id)

            next_global_id += 1

            temp_assignments[tid] = gid
            temp_new_flags[tid]   = True

            log_match(
                f"[ASSIGN] TID={tid} -> NEW GID={gid} "
                f"(compared against {len(REID_DICT)} stored GID(s), no match)"
            )

        elif len(matched) > 1:

            temp_assignments[tid] = None
            temp_new_flags[tid]   = False

            log_match(
                f"[ASSIGN] TID={tid} -> CONFLICT, matched multiple GIDs={matched}, "
                f"left unassigned this frame"
            )

        else:

            temp_assignments[tid] = int(matched[0])
            temp_new_flags[tid]   = False

            log_match(
                f"[ASSIGN] TID={tid} -> MATCHED existing GID={matched[0]}"
            )

    # =========================================================
    # PASS 2: RESOLVE CONFLICTS
    # =========================================================

    gid_to_tids = {}

    for tid, gid in temp_assignments.items():

        if gid is not None:

            gid = int(gid)

            gid_to_tids.setdefault(gid, []).append(tid)

    final_assignments = {
        tids[0]: int(gid)
        for gid, tids in gid_to_tids.items()
        if len(tids) == 1
    }

    # =========================================================
    # HELPER: SPLIT GALLERY
    # =========================================================

    def split_gallery(gallery):

        recent = sorted(
            gallery,
            key=lambda x: x[2],
            reverse=True
        )[:M]

        conf = sorted(
            gallery,
            key=lambda x: x[1],
            reverse=True
        )[:N]

        sharp = sorted(
            gallery,
            key=lambda x: x[6],
            reverse=True
        )[:L]

        return recent, conf, sharp

    # =========================================================
    # HELPER: DEDUPLICATE GALLERY
    # =========================================================

    def deduplicate_gallery(gallery):

        unique_gallery = []
        seen_frames    = set()

        for item in gallery:

            fid = item[2]

            if fid in seen_frames:
                continue

            seen_frames.add(fid)

            unique_gallery.append(item)

        return sorted(
            unique_gallery,
            key=lambda x: x[2]
        )

    # =========================================================
    # PASS 3: APPLY + MERGE
    # =========================================================

    track_to_global = {}

    for tid, gid in final_assignments.items():

        gid = int(gid)

        new_gallery = temp_galleries[tid]

        # =========================================================
        # NEW GLOBAL ID
        # =========================================================

        if gid not in REID_DICT:

            REID_DICT[gid] = {

                "gallery": new_gallery,

                "baseline": 
                    (
                        self_baseline_score(new_gallery)
                        if USE_VISUAL_REID 
                        else 0.0
                    ),
            }
            
            save_gallery_crops(gid, new_gallery, tid, frame_id, event="new")

            MISSION_STATS["gids_created"] += 1
            log_info(f"Saving new GID {gid}")


            save_new_gid_artifacts(
                gid,
                new_gallery,
                ros_time,
                event="new",
                tid=tid,
                buffer_gallery=new_gallery,
            )

        # =========================================================
        # MERGE EXISTING GALLERY
        # =========================================================

        else:

            old_gallery = REID_DICT[gid]["gallery"]

            old_r, old_c, old_s = split_gallery(old_gallery)
            new_r, new_c, new_s = split_gallery(new_gallery)

            merged_recent = sorted(
                new_r,
                key=lambda x: x[2]
            )

            merged_conf = sorted(
                sorted(
                    old_c + new_c,
                    key=lambda x: x[1],
                    reverse=True
                )[:N],
                key=lambda x: x[2]
            )

            merged_sharp = sorted(
                sorted(
                    old_s + new_s,
                    key=lambda x: x[6],
                    reverse=True
                )[:L],
                key=lambda x: x[2]
            )

            merged_gallery = (
                merged_recent
                + merged_conf
                + merged_sharp
            )

            # merged_gallery = deduplicate_gallery(
            #     merged_gallery
            # )

            REID_DICT[gid] = {

                "gallery": merged_gallery,

                "baseline": 
                    (
                        self_baseline_score(merged_gallery)
                        if USE_VISUAL_REID 
                        else 0.0
                    ),
            }

            save_gallery_crops(gid, merged_gallery, tid, frame_id, event="merge")

            save_new_gid_artifacts(
                gid,
                merged_gallery,
                ros_time,
                event="merge",
                tid=tid,
                buffer_gallery=new_gallery,
            )

            MISSION_STATS["gallery_merges"] += 1

        # =========================================================
        # UPDATE moving_tracks
        # =========================================================

        final_gallery = REID_DICT[gid]["gallery"]

        moving_tracks[tid] = [
            [*item, gid, tid]
            for item in final_gallery
        ]

        track_to_global[tid] = gid

        # =========================================================
        # GPS DEBUG STORAGE
        # =========================================================

        gps_points = []

        for item in new_gallery:

            lat = item[7]
            lon = item[8]

            if lat is not None and lon is not None:

                gps_points.append((lat, lon))

        if len(gps_points) > 0:

            medoid_lat, medoid_long = robust_gallery_medoid(gps_points)

            GPS_DEBUG.setdefault(gid, []).append({

                "tid": tid,

                "gps_points": gps_points,

                "medoid_lat": medoid_lat,
                "medoid_long": medoid_long
            })

        # =========================================================
        # GPS VARIANCE DEBUG
        # =========================================================

        if len(gps_points) >= 2:

            distances = []

            for i in range(1, len(gps_points)):

                d = haversine_distance(
                    gps_points[i-1][0],
                    gps_points[i-1][1],
                    gps_points[i][0],
                    gps_points[i][1]
                )

                distances.append(d)

            mean_jump = np.mean(distances)
            max_jump  = np.max(distances)

            log_gps(
                f"[GPS STABILITY] "
                f"GID={gid} "
                f"TID={tid} "
                f"POINTS={len(gps_points)} "
                f"MEAN_JUMP={mean_jump:.2f}m "
                f"MAX_JUMP={max_jump:.2f}m "
                f"GALLERY_MEDOID=({medoid_lat:.6f}, {medoid_long:.6f})"
            )

        # =========================================================
        # DEBUG GPS OUTPUT (MEDOID)
        # =========================================================

        gps_points = [
            (item[7], item[8])
            for item in final_gallery
            if item[7] is not None and item[8] is not None
        ]

        if gps_points:

            medoid_lat, medoid_lon = robust_gallery_medoid(gps_points)

        else:

            medoid_lat = None
            medoid_lon = None

        log_gps(
            f"[GPS ASSIGNED] "
            f"GID={gid} "
            f"TID={tid} "
            f"MEDOID_LAT={medoid_lat} "
            f"MEDOID_LON={medoid_lon}"
        )

    return track_to_global



# =========================================================
# MODULE 11: MAIN PIPELINE LOOP
# ================================================

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    RUN_FOLDER = os.path.join(
        BASE_DIR,
        "mission_logs",
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    os.makedirs(RUN_FOLDER, exist_ok=True)

    if SAVE_GALLERIES:
        # Use configured path if set, otherwise default to RUN_FOLDER/gallery_dumps
        if GALLERY_SAVE_DIR is None:
            GALLERY_SAVE_DIR = os.path.join(RUN_FOLDER, "gallery_dumps")
        os.makedirs(GALLERY_SAVE_DIR, exist_ok=True)
        log_info(f"[GALLERY] Saving to: {GALLERY_SAVE_DIR}")

    print("=" * 80)
    print("DEBUG STARTUP")
    print("RUN_FOLDER :", RUN_FOLDER)
    print("ABS PATH   :", os.path.abspath(RUN_FOLDER))
    print("EXISTS     :", os.path.exists(RUN_FOLDER))
    print("CWD        :", os.getcwd())
    print("=" * 80)
    
    log_info(f"Mission Folder: {os.path.abspath(RUN_FOLDER)}")

    LOG_FILE = os.path.join(
        RUN_FOLDER,
        "mission.log"
    )

    with open(LOG_FILE, "w") as f:
        f.write("="*80 + "\n")
        f.write("SEARCH & RESCUE PIPELINE\n")
        f.write(datetime.now().isoformat() + "\n")
        f.write("="*80 + "\n\n")

    print(f"[RUN FOLDER] {RUN_FOLDER}")

    log_banner("SEARCH & RESCUE PIPELINE")

    log_info(f"CUDA Available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():

        log_info(f"GPU : {torch.cuda.get_device_name(0)}")

    log_info(f"ONNX Runtime : {ort.__version__}")

    log_info(f"Mission Folder : {RUN_FOLDER}")

    log_section("Loading Models")

    # =========================================================
    # DETECTOR (ALWAYS REQUIRED)
    # =========================================================

    detector = load_detector()

    if not USE_ROBUST_DETECT:

        # ---- warmup so Ultralytics actually builds the predictor/session ----
        _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.predict(_dummy, imgsz=YOLOE_IMGSZ, conf=YOLOE_CONF,
                         iou=YOLOE_IOU, device=YOLOE_DEVICE, verbose=False)

        log_info(f"ORT available providers : {ort.get_available_providers()}")
        try:
            _sess = detector.predictor.model.session
            log_info(f"Detector session providers : {_sess.get_providers()}")
        except Exception as e:
            log_warn(f"Could not reach detector session: {e}")

    log_ok(f"YOLOE Weights : {YOLOE_WEIGHTS}")

    if USE_ROBUST_DETECT:

        # The robust path runs its own prompt lists, not the pipeline's,
        # so print what is actually loaded rather than POSITIVE_CLASSES
        # / NEGATIVE_CLASSES, which this mode never reads.
        log_ok(f"YOLOE Classes (positive) : {detector.positives}")
        log_ok(f"YOLOE Classes (negative) : {detector.negatives}")

    else:

        log_ok(f"YOLOE Classes (positive) : {POSITIVE_CLASSES}")
        log_ok(f"YOLOE Classes (negative) : {NEGATIVE_CLASSES}")

    if USE_ROBUST_DETECT:

        _tiles = ROBUST_REF.plan_tiles(
            1920, 1080,
            detector.tile_scale,
            detector.tile_overlap,
            YOLOE_IMGSZ,
            detector.tile_max,
        )

        log_ok(
            f"Detector : ROBUST ({ROBUST_MOD.__name__}, "
            f"preset={ROBUST_PRESET}) - {len(_tiles)} tiles/frame "
            f"+ full frame, WBF fusion, photometric shadow gate"
        )

        # YOLOE_CONF belongs to the legacy path only; say so rather than
        # printing a threshold this mode never reads.
        log_ok(
            f"YOLOE Params  : tile-floor={detector.conf_floor} "
            f"min_conf={getattr(ROBUST_REF, 'MIN_CONF', 0.0)} "
            f"shadow_reject={detector.shadow_reject} "
            f"imgsz={YOLOE_IMGSZ} device={YOLOE_DEVICE} "
            f"(YOLOE_CONF={YOLOE_CONF} unused in this mode)"
        )

        if ROBUST_LABEL == "cuda":
            log_ok(
                f"CUDA fast path : precision={detector.precision} "
                f"backend={detector.backend} workers={ROBUST_WORKERS}"
            )

    else:

        log_ok("Detector : legacy single-shot (USE_ROBUST_DETECT=False)")
        log_ok(
            f"YOLOE Params  : conf={YOLOE_CONF} "
            f"iou={YOLOE_IOU} "
            f"imgsz={YOLOE_IMGSZ} "
            f"device={YOLOE_DEVICE}"
        )

    # =========================================================
    # OPTIONAL VISUAL REID
    # =========================================================

    reid_model = None

    if USE_VISUAL_REID:
        reid_model = load_reid()
        log_ok(f"ReID Backend : {reid_model.backend}")

        print("\n=================================================")
        print("WEIGHTED GALLERY REID ENABLED")
        print("=================================================")

        print(f"RATIO_THRESHOLD         = {RATIO_THRESHOLD}")
        print(f"MIN_SUPPORT_CONFIDENCE  = {MIN_SUPPORT_CONFIDENCE}")
        print(f"TOP_K_DIMS              = {TOP_K_DIMS}")
        print(f"SUPPORT_TOP_K           = {SUPPORT_TOP_K}")

        print("=================================================\n")

    log_ok("Models loaded successfully.")

    # =========================================================
    # TELEMETRY + VIDEO SOURCE INIT
    # =========================================================

    if TELEM_SOURCE == "csv":
        load_csv_telemetry(TELEM_CSV_PATH)
        log_info(f"[TELEM] CSV replay: {TELEM_CSV_PATH}")

        # Auto-derive VIDEO_START_UNIX from CSV's first wall-clock entry
        if VIDEO_START_UNIX is None:
            import csv as _csv
            from datetime import datetime as _dt
            # Same encoding guards as load_csv_telemetry(): this reads
            # the very same file, so a BOM or a stray non-UTF-8 byte
            # would crash here instead, one block later.
            with open(TELEM_CSV_PATH, encoding="utf-8-sig",
                      errors="replace", newline="") as _f:
                _r = next(_csv.DictReader(_f))
                VIDEO_START_UNIX = _dt.fromisoformat(
                    _r["t_wall_utc"].strip()
                ).timestamp()
            log_info(f"[TELEM] Auto-derived VIDEO_START_UNIX = {VIDEO_START_UNIX}")

    elif TELEM_SOURCE == "live":
        log_info("[TELEM] Live Giver/ROS")
    else:
        raise ValueError(f"Invalid TELEM_SOURCE: {TELEM_SOURCE!r}")

    init_video_source()

    # =========================================================
    # OUTPUT DISPLAY
    # =========================================================

    if DISPLAY_MODE == "stream":
        start_stream_server(STREAM_PORT)
        log_ok(f"[DISPLAY] MJPEG stream on http://<this-host>:{STREAM_PORT}/")

    elif DISPLAY_MODE == "window":
        log_ok("[DISPLAY] cv2.imshow window (ESC to quit)")

    else:
        log_ok("[DISPLAY] none - running headless")

    # =========================================================
    # VIDEO
    # =========================================================

    # Absolute index into the video, NOT a count of frames processed, so
    # frame_id / _video_fps stays the true elapsed time and the CSV
    # lookup in read_source() lines up with whatever START_FRAME asked
    # for. At START_FRAME=0 this is 0 and nothing changes.
    frame_id        = _start_frame_actual
    track_to_global = {}
    prev_time       = 0

    MISSION_START_ROS_TIME = None

    # =========================================================
    # MAIN LOOP
    # =========================================================

    while True:

        loop_start = time.perf_counter()

        ret, frame, telem = read_source(frame_id + 1)

        if not ret:
            if VIDEO_SOURCE == "video":
                break     # end of file
            time.sleep(0.005)
            continue

        ros_time = telem.get("timestamp_sec") if telem else None

        if MISSION_START_ROS_TIME is None and ros_time is not None:
            MISSION_START_ROS_TIME = ros_time

        # Snapshot telem at this exact frame so all detections
        # in this frame use the same synchronized telem data.
    

        # =========================================================
        # FPS
        # =========================================================

        curr_time = time.time()

        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0.0

        prev_time = curr_time

        log_frame(f"FPS={fps:.1f}")

        frame_id += 1

        MISSION_STATS["frames"] += 1

        MISSION_STATS["fps_sum"] += fps

        MISSION_STATS["fps_min"] = min(
            MISSION_STATS["fps_min"],
            fps
        )

        MISSION_STATS["fps_max"] = max(
            MISSION_STATS["fps_max"],
            fps
        )

        frame_timestamp = time.time()

        h, w = frame.shape[:2]

        # =========================================================
        # STEP 1: DETECTION
        # =========================================================

        # YOLOE already applies its own NMS (YOLOE_IOU) internally,
        # so its boxes go straight to feature extraction.
        # Physical-size gate: metres per pixel for THIS frame.
        if USE_ROBUST_DETECT and ROBUST_LABEL == "cuda":
            _alt = telem.get("altitude") if telem else None
            # Same per-frame zoom the projection uses, so the physical-size
            # gate and the geolocation never disagree about how wide a pixel is.
            detector.gsd = (_alt / (CAMERA_INTRINSICS[0, 0] * telem_zoom(telem))
                            if _alt else None)

        boxes = detect(detector, frame)

        MISSION_STATS["detections_raw"] += len(boxes)

        # Remove person-like boxes that overlap shadow boxes,
        # then strip the class name so downstream code stays unchanged.
        boxes = suppress_shadow_overlaps(boxes)
        boxes = [b[:5] for b in boxes]   # drop cls_name → back to (x1,y1,x2,y2,conf)

        # =========================================================
        # STEP 2: FEATURE EXTRACTION
        # =========================================================

        valid_boxes, embeddings = prepare_detections_and_embeddings(
            frame,
            boxes,
            reid_model,
            frame_id,
            frame_timestamp,
            telem
        )

        MISSION_STATS["detections_final"] += len(valid_boxes)

        # =========================================================
        # STEP 3: IOU TRACK ASSIGNMENT
        # =========================================================

        assign_detections_to_tracks(
            valid_boxes,
            embeddings,
            frame_id
        )

        # =========================================================
        # STEP 4: BUFFER PROCESSING
        # =========================================================

        reid_candidates = process_track_buffers(frame_id)

        # =========================================================
        # STEP 5: GLOBAL ASSOCIATION
        # =========================================================

        new_assignments = process_reid_candidates(
            reid_candidates,
            ros_time,
            frame_id
        )

        track_to_global.update(new_assignments)

        # =========================================================
        # STEP 6: VISUALIZATION
        # =========================================================

        # =========================================================
        # CENTRAL FOV DEBUG VISUALIZATION
        # =========================================================

        if USE_CENTER_FOV_FILTER:

            valid_w = int(w * CENTER_FOV_RATIO)
            valid_h = int(h * CENTER_FOV_RATIO)

            valid_x1 = int((w - valid_w) / 2)
            valid_y1 = int((h - valid_h) / 2)

            valid_x2 = valid_x1 + valid_w
            valid_y2 = valid_y1 + valid_h

            cv2.rectangle(
                frame,
                (valid_x1, valid_y1),
                (valid_x2, valid_y2),
                (255, 0, 0),
                2
            )

        # =========================================================
        # DRAW DETECTIONS
        # =========================================================

        for (x1, y1, x2, y2, conf, _, lat, lon, _) in valid_boxes:

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            display_tid = None

            for tid, hist in tracks.items():

                last = hist[-1]

                if (
                    abs(last[3] - cx) < 1e-3
                    and
                    abs(last[4] - cy) < 1e-3
                ):
                    display_tid = tid
                    break

            display_gid = (
                track_to_global.get(display_tid)
                if display_tid is not None else None
            )

            # =====================================================
            # BBOX
            # =====================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # =====================================================
            # LABEL
            # =====================================================

            text = f"C:{conf:.2f} T:{display_tid}"

            if display_gid is not None:
                text += f" G:{display_gid}"

            # =====================================================
            # OPTIONAL GPS DISPLAY
            # =====================================================

            if display_gid is not None:

                gallery = REID_DICT.get(display_gid, {}).get("gallery", [])

                gps_points = [
                    (item[7], item[8])
                    for item in gallery
                    if item[7] is not None and item[8] is not None
                ]

                if gps_points:

                    medoid_lat, medoid_lon = robust_gallery_medoid(gps_points)

                    text += f" ({medoid_lat:.5f}, {medoid_lon:.5f})"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # =========================================================
        # STEP 7: CASUALTY COUNT
        # =========================================================

        cv2.putText(
            frame,
            f"Total Casualties: {len(REID_DICT)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # =========================================================
        # DISPLAY
        # =========================================================
        if DISPLAY_MODE == "window":

            cv2.imshow("Live UAV Casualty Tracking", frame)

            # waitKey is what pumps OpenCV's GUI event loop; without a
            # call to it the window is created but never paints. It also
            # returns the keypress the ESC check below reads.
            key = cv2.waitKey(1) & 0xFF

        elif DISPLAY_MODE == "stream":

            # Hand the frame to the server thread. .copy() because the
            # loop draws over `frame` in place on the next iteration,
            # and the encoder may still be reading this one.
            with _frame_lock:
                _output_frame = frame.copy()
                _output_seq  += 1

            key = 0

        else:                                   # "none"

            key = 0

        if VIDEO_SOURCE == "video" and _frame_time is not None:
            elapsed = time.perf_counter() - loop_start
            if elapsed < _frame_time:
                time.sleep(_frame_time - elapsed)

        if key == 27:
            break

    # =========================================================
    # CLEANUP
    # =========================================================

    runtime = time.time() - MISSION_STATS["start_time"]

    log_banner("MISSION SUMMARY")

    log_info(f"Runtime              : {runtime:.1f} sec")
    log_info(f"Frames               : {MISSION_STATS['frames']}")

    if MISSION_STATS["frames"]:

        log_info(
            f"Average FPS          : "
            f"{MISSION_STATS['fps_sum']/MISSION_STATS['frames']:.2f}"
        )

    log_info(f"Minimum FPS          : {MISSION_STATS['fps_min']:.2f}")
    log_info(f"Maximum FPS          : {MISSION_STATS['fps_max']:.2f}")

    log_info(f"Raw detections       : {MISSION_STATS['detections_raw']}")
    log_info(f"Final detections     : {MISSION_STATS['detections_final']}")

    log_info(f"FOV rejects          : {MISSION_STATS['fov_rejects']}")
    log_info(f"Blur rejects         : {MISSION_STATS['blur_rejects']}")
    log_info(f"Shadow rejects       : {MISSION_STATS['shadow_rejects']}")
    log_info(f"Invalid rejects      : {MISSION_STATS['invalid_rejects']}")
    log_info(f"GPS failures         : {MISSION_STATS['gps_failures']}")

    log_info(f"Tracks created       : {MISSION_STATS['tracks_created']}")
    log_info(f"Tracks deleted       : {MISSION_STATS['tracks_deleted']}")

    log_info(f"GIDs created         : {MISSION_STATS['gids_created']}")
    log_info(f"Gallery merges       : {MISSION_STATS['gallery_merges']}")

    log_info(f"Final Casualties     : {len(REID_DICT)}")

    log_info(f"Mission Folder       : {RUN_FOLDER}")

    # Robust detector's own funnel: how many raw boxes the tiles
    # produced and where each one was lost (seam / fusion / shadow /
    # geometry), plus the per-stage millisecond split on the CUDA path.
    if USE_ROBUST_DETECT:
        try:
            log_info(detector.report())
            if hasattr(detector, "timing_report"):
                log_info(detector.timing_report())
            if hasattr(detector, "close"):
                detector.close()
        except Exception as e:
            log_warn(f"Could not report detector stats: {e}")

    if VIDEO_SOURCE == "live" and giver is not None:
        giver.shutdown()
    else:
        try:
            if _cap is not None:
                _cap.release()
        except NameError:
            pass

    if DISPLAY_MODE == "window":
        cv2.destroyAllWindows()

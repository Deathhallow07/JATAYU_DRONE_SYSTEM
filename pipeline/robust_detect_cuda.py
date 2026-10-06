"""
=========================================================
ROBUST PERSON DETECTION - CUDA FAST PATH (DGX SPARK)
=========================================================

Same detector as robust_detect.py. Same tiles, same seam rule, same
weighted box fusion, same photometric shadow gate, same constants.
Only the plumbing is different.

The problem this file solves
----------------------------
On a DGX Spark (GB10: 20 Arm cores + Blackwell GPU + 128 GB unified
LPDDR5X) robust_detect.py ran at 3 FPS with one CPU core pinned at
~100% and almost no VRAM in use. That combination is diagnostic. The
GPU was not the bottleneck - it was idle, waiting on a single Python
thread to hand it work. A 26s-sized network at batch 10 cannot
saturate a Blackwell; if it is slow, the time is going somewhere
other than the math.

It was. Measured per frame on the original path (1080p, 9 tiles + 1
full frame). The absolute numbers below are from an Arc XPU used as a
stand-in; the DGX Spark's split is worse, because its per-core CPU
speed is lower and its GPU is faster, so the non-GPU lines weigh more:

    total detect()                            490 ms
      model forward, fp32 640x640, with proto 336 ms
      ultralytics predict() overhead          112 ms
      cv2 letterbox of 10 images + H2D         42 ms
      photometric shadow scoring (5 boxes)     25 ms

Every one of those four lines is mostly waste.

The five fixes, and what each is worth
--------------------------------------
1.  THE MASK HEAD WAS BEING COMPUTED AND THROWN AWAY.
    yoloe-26s-seg is a segmentation model. Its Proto branch builds a
    (B, 32, 96, 160) prototype tensor on every forward, and
    ultralytics then runs process_mask over every surviving box. This
    pipeline reads r.boxes and nothing else - no mask was ever looked
    at. Skipping the Proto branch is bit-identical on the detections
    and is the single largest win in this file:

        forward with proto      276 ms
        forward without proto   152 ms      max box diff 0.0

2.  SQUARE LETTERBOX WASTED 40% OF EVERY FORWARD.
    A 16:9 frame letterboxed into 640x640 is 640x360 of image and 280
    rows of grey padding - 44% of the tensor is a constant. The
    network is fully convolutional and stride-32, so 640x384 holds
    the identical image at the identical scale with 12 rows of pad
    top and bottom. Same pixels in, 40% fewer to convolve.

3.  PREPROCESSING BELONGS ON THE GPU.
    The old path did, on the CPU, for ten images per frame: crop,
    cv2.resize, copyMakeBorder, BGR->RGB, transpose,
    ascontiguousarray, then a pageable host-to-device copy. That is
    ~14 MB of memcpy and ten resizes on one Arm core. Here the frame
    is uploaded ONCE (6 MB, through a pinned buffer) and every tile is
    a stride view of that GPU tensor; all of the crops are resized by
    a single batched F.interpolate into one preallocated padded
    canvas. Host cost per frame falls to a single 6 MB memcpy.

4.  fp16 + channels_last. Blackwell's tensor cores want both.
    152 ms -> 87 ms, worst-case confidence delta 0.0012.

5.  CUDA GRAPHS. This is the fix aimed squarely at the 102% CPU.
    A small network is hundreds of tiny kernels, and on an Arm host
    the launch cost of each one is a larger fraction of its runtime
    than it would be on x86. Capturing preprocessing plus the whole
    forward into one graph collapses every launch into a single
    replay: the CPU issues one call per frame and the GPU runs the
    recorded sequence with no host in the loop.

    Two things in ultralytics block capture. Both are patched on the
    head INSTANCE here - nothing global is monkeypatched, so importing
    this module cannot change another model's behaviour:
      - Detect.get_topk_index builds torch.arange(batch) on the CPU
        and indexes a CUDA tensor with it. That is a pageable H2D copy
        mid-graph, which is not capturable. _graph_safe_topk is the
        same computation with a device-resident arange, verified
        bit-identical below.
      - YOLOESegment26.forward calls the Proto branch, which fix 1
        removes anyway.
    If capture fails for any reason the detector says so once and
    falls back to eager. It never silently produces different boxes.

Two smaller ones
----------------
6.  head.max_det 300 -> 150. The end2end head emits a fixed top-k per
    image whether or not there is anything to report, so the tail is
    pure cost. At conf floor 0.06 this footage yields 2.6 boxes per
    image; 150 is a 50x margin and halves both the gather and the
    device-to-host transfer.
7.  The shadow gate runs in a thread pool. It is numpy and OpenCV,
    both of which release the GIL, so on 20 Arm cores the per-frame
    cost becomes the slowest single box rather than their sum.

What this does NOT change
-------------------------
The detections. plan_tiles, the seam rule, weighted_box_fusion,
confidence_adjust, the seven shadow features and their fitted
coefficients are all imported unchanged from robust_detect, so there
is exactly one copy of the physics and the tuning constants.
Refitting with fit_shadow_model.py still lands in both files at once.

The only intended numeric differences are fp16 arithmetic in the
backbone (~1e-3 on confidence) and bilinear resampling done by torch
rather than by cv2. `--verify` measures both against the original
detector on real frames and prints the agreement, so neither has to
be taken on faith.

Usage
-----
    from robust_detect_cuda import make_detector
    det = make_detector("balanced")          # picks CUDA automatically
    for d in det.detect(frame):
        print(d.x1, d.y1, d.x2, d.y2, d.conf, d.person_score, d.prompt)

Drop-in for the old module: RobustPersonDetector, Detection,
detect_boxes and make_detector keep their names and signatures, so

    -import robust_detect as RD
    +import robust_detect_cuda as RD

is the whole migration.

Command line
------------
    python robust_detect_cuda.py --source test.mp4 --bench
    python robust_detect_cuda.py --source test.mp4 --verify 20
    python robust_detect_cuda.py --source test.mp4 --save out.mp4

Tuning on the Spark, in the order worth trying
----------------------------------------------
    --backend graph      (default) one replay per frame
    --precision fp16     (default) bf16 if fp16 ever looks unstable
    --preset fast        4 tiles instead of 9, roughly 2x the rate
    --imgsz 512          only if targets are large; 640 is the tuned value
    --workers 8          shadow-gate threads
    --reader-threads 2   decode ahead of the GPU
"""

from __future__ import annotations

import argparse
import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLOE
from ultralytics.nn.modules.head import YOLOEDetect

# Single source of truth for geometry, fusion and physics. Nothing
# here re-derives a tuned constant.
from robust_detect import (
    CONF_FLOOR,
    Detection,
    ENABLE_TILING,
    IMGSZ,
    MAX_ASPECT,
    MAX_BOX_FRAC,
    MIN_BOX_PX,
    NEGATIVE_PROMPTS,
    POSITIVE_PROMPTS,
    RING_CAP_PX,
    RING_FRAC,
    RING_MIN_PX,
    SEAM_MARGIN,
    SHADOW_BIAS,
    SHADOW_COEF,
    SHADOW_CONF_WEIGHT,
    SHADOW_FEATURES,
    SHADOW_MEAN,
    SHADOW_REJECT,
    SHADOW_SCALE,
    TILE_MAX,
    TILE_OVERLAP,
    TILE_TARGET_SCALE,
    WEIGHTS,
    PERSON_MAX_M,
    PERSON_MIN_M,
    confidence_adjust,
    person_score as person_score_reference,
    plan_tiles,
    size_verdict,
    weighted_box_fusion,
)


# =========================================================
# CUDA CONFIGURATION
# =========================================================

# Per-image cap out of the end2end head (see fix 6 above).
MAX_DET = 150

# Network canvas. RECT_INPUT letterboxes into a stride-32 box matching
# the frame's aspect instead of a square: 640x384 for 16:9, the same
# image at the same scale with 40% fewer pixels. Set False to
# reproduce the original square geometry exactly.
RECT_INPUT = True
STRIDE = 32
PAD_VALUE = 114 / 255.0     # ultralytics' letterbox grey

# Downscaling 1920 -> 640 is a 3x reduction, which plain bilinear
# aliases. cv2.INTER_LINEAR, which the original used, aliases the same
# way, so OFF is the parity-preserving default. Turning it on is a
# quality option, not a speed one: it costs about a millisecond and it
# changes what the full-frame pass sees.
ANTIALIAS = False

# Threads for the photometric gate. 0 means one per detection, capped.
SHADOW_WORKERS = 8

# OpenCV's own parallel_for pool size, set once when a detector is
# built. This is the single most important number in this file for
# real streaming throughput, and it is not obvious why.
#
# Decoding runs on the reader thread, and OpenCV decodes with its
# thread pool -- 16 workers on this box. The photometric gate runs
# cvtColor and Laplacian from SHADOW_WORKERS threads of its own, and
# those go through the same pool. Overlap the two, as any real
# pipeline does, and the machine is oversubscribed several times over;
# the CPU spends its time in the scheduler, and the thread that
# actually matters -- the one issuing CUDA work -- is descheduled
# along with everything else.
#
# Measured, 150 frames at 1080p, threaded reader, same detector:
#
#     cv2 threads 16   152.9 ms/frame    6.5 FPS
#     cv2 threads  1    39.6 ms/frame   25.2 FPS
#
# That is a 3.9x swing with no change to a single detection. Nothing
# here benefits from the pool anyway: the gate's images are ~300x300
# crops, far below the size where splitting one across 16 cores beats
# the cost of dispatching it, and decode is 1.6 ms/frame either way,
# on its own thread, hidden behind a 40 ms forward pass.
#
# Set to 0 to leave OpenCV's default alone.
CV_THREADS = 1

# How close to the frame border counts as "touching it", for the size
# gate in detect(). A box this near the edge is assumed to be cut by
# it, so its width says nothing reliable about the target's extent.
EDGE_MARGIN = 2

DEFAULT_BACKEND = "graph"       # graph | eager | compile
DEFAULT_PRECISION = "fp16"      # fp16 | bf16 | fp32

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# log(v + 2.0) for every uint8 value. shadow_features spends a
# meaningful slice of its time in np.log over the context patch; a
# 256-entry lookup is the identical computation on identical inputs,
# so this is free accuracy-wise.
_LOG_LUT = np.log(np.arange(256, dtype=np.float32) + 2.0)


def _tune_torch():
    """Backend switches that only ever help a fixed-shape vision graph."""
    torch.backends.cudnn.benchmark = True          # shapes never change
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _tune_opencv(n=None):
    """
    Stop OpenCV oversubscribing the CPU. See CV_THREADS for the
    measurement; this is a process-wide setting, so it is called when
    a detector is built rather than at import, and a caller that has
    its own opinion can pass n=0 to leave it untouched.
    """
    n = CV_THREADS if n is None else n
    if n:
        cv2.setNumThreads(int(n))


def pick_device(requested=None):
    """CUDA first in this file - that is the point of it."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0")
    return torch.device("cpu")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


# =========================================================
# LETTERBOX PLAN
# =========================================================

def _round_stride(v, stride=STRIDE):
    return max(stride, int(math.ceil(v / stride) * stride))


@dataclass
class LetterboxPlan:
    """
    Where one source size lands inside the network canvas.

    Uniform scale, centred pad - exactly what ultralytics' LetterBox
    does, just expressed as numbers instead of a cv2 call, so the same
    arithmetic can run on the GPU and be inverted on the way out.
    """
    src_w: int
    src_h: int
    new_w: int
    new_h: int
    pad_x: int
    pad_y: int
    scale: float

    @classmethod
    def build(cls, src_w, src_h, canvas_w, canvas_h):
        r = min(canvas_w / src_w, canvas_h / src_h)
        nw, nh = int(round(src_w * r)), int(round(src_h * r))
        nw, nh = min(nw, canvas_w), min(nh, canvas_h)
        return cls(src_w, src_h, nw, nh,
                   (canvas_w - nw) // 2, (canvas_h - nh) // 2, r)

    @property
    def offset(self):
        """Subtract from a canvas-space xyxy box before dividing by scale."""
        return np.array([self.pad_x, self.pad_y, self.pad_x, self.pad_y],
                        np.float32)


@dataclass
class FramePlan:
    """Everything about one input resolution, computed once and cached."""
    w: int
    h: int
    canvas_w: int
    canvas_h: int
    tiles: list
    full: LetterboxPlan
    tile: LetterboxPlan | None
    tile_w: int = 0
    tile_h: int = 0
    origins: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    interior: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), bool))

    @property
    def batch(self):
        return 1 + len(self.tiles)


def build_plan(w, h, imgsz=IMGSZ, tiling=ENABLE_TILING,
               tile_scale=TILE_TARGET_SCALE, tile_overlap=TILE_OVERLAP,
               tile_max=TILE_MAX, rect=RECT_INPUT):
    """
    Tile rectangles plus the letterbox arithmetic for one resolution.

    plan_tiles is imported, so the grid is identical to the original
    detector's. It guarantees every tile is exactly tile_w x tile_h
    (the last row and column are pulled flush rather than truncated),
    which is what lets all of them share one batched resize.
    """
    tiles = plan_tiles(w, h, tile_scale, tile_overlap, imgsz, tile_max) if tiling else []

    if rect:
        if w >= h:
            canvas_w = imgsz
            canvas_h = min(imgsz, _round_stride(imgsz * h / w))
        else:
            canvas_h = imgsz
            canvas_w = min(imgsz, _round_stride(imgsz * w / h))
    else:
        canvas_w = canvas_h = imgsz

    plan = FramePlan(w=w, h=h, canvas_w=canvas_w, canvas_h=canvas_h,
                     tiles=tiles,
                     full=LetterboxPlan.build(w, h, canvas_w, canvas_h),
                     tile=None)

    if tiles:
        tw = tiles[0][2] - tiles[0][0]
        th = tiles[0][3] - tiles[0][1]
        # Cheap insurance: a ragged grid would silently corrupt the
        # batched resize, so fall back to no tiling rather than guess.
        if any((t[2] - t[0]) != tw or (t[3] - t[1]) != th for t in tiles):
            plan.tiles = []
            return plan
        plan.tile_w, plan.tile_h = tw, th
        plan.tile = LetterboxPlan.build(tw, th, canvas_w, canvas_h)
        plan.origins = np.array([[t[0], t[1]] for t in tiles], np.float32)
        # Which edges of each tile are cuts through the frame rather
        # than the frame's own border. Only a cut can orphan a target.
        plan.interior = np.array(
            [[t[0] > 0, t[1] > 0, t[2] < w, t[3] < h] for t in tiles], bool)
    return plan


# =========================================================
# PHOTOMETRIC SHADOW / CLUTTER SCORE  (fast port)
# =========================================================

def shadow_features_fast(frame, box, ring_cap=RING_CAP_PX, ring_frac=RING_FRAC):
    """
    Line-for-line the same seven features as
    robust_detect.shadow_features - read that docstring for what each
    one measures and why. Two mechanical changes, no formula changes:

      * np.log over the context patch becomes a 256-entry lookup.
        Identical inputs, identical op, identical floats.
      * The ring mask is flattened once into an index array instead of
        being applied as a boolean mask five separate times.

    `--verify` asserts this against the original on real boxes.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 6 or bh < 6:
        return None

    pad = int(max(bw, bh) * ring_frac)
    pad = max(RING_MIN_PX, min(pad, ring_cap))
    X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
    X2, Y2 = min(w, x2 + pad), min(h, y2 + pad)
    ctx = np.ascontiguousarray(frame[Y1:Y2, X1:X2])
    if ctx.size == 0:
        return None

    ox1, oy1 = x1 - X1, y1 - Y1
    ox2, oy2 = ox1 + bw, oy1 + bh
    ring = np.ones(ctx.shape[:2], bool)
    ring[oy1:oy2, ox1:ox2] = False
    if ring.sum() < 80:
        return None
    ridx = np.flatnonzero(ring.ravel())

    L = _LOG_LUT[ctx]                       # == np.log(ctx.astype(f32) + 2.0)
    LC = L - L.mean(2, keepdims=True)

    ring_lc = LC.reshape(-1, 3)[ridx]
    bg = np.median(ring_lc, 0)
    spread = ring_lc.std(0).mean() + 1e-3

    inside = LC[oy1:oy2, ox1:ox2]
    d = np.linalg.norm(inside - bg, axis=2) / (spread * 3.0)

    hsv = cv2.cvtColor(ctx, cv2.COLOR_BGR2HSV)
    hh = hsv[..., 0].astype(np.float32)
    ss = hsv[..., 1].astype(np.float32)
    vv = hsv[..., 2].astype(np.float32)

    # One gather for the whole ring instead of one per channel. ridx
    # is the same index array either way, so these are the same
    # values; it is the three separate fancy-index passes over a
    # ~40k-element array that were costing something.
    ring_hsv = hsv.reshape(-1, 3)[ridx].astype(np.float32)
    h_rg, s_rg, v_rg = ring_hsv[:, 0], ring_hsv[:, 1], ring_hsv[:, 2]

    v_in = vv[oy1:oy2, ox1:ox2]
    # np.percentile sorts (partitions) its input once per call, and
    # the 5th, 50th and 95th were three calls over the same array --
    # three partitions where one does. np.median(x) is by definition
    # np.percentile(x, 50) with linear interpolation, including the
    # even-length average, so this is the same number, not a near one.
    q_in = np.percentile(v_in, (5.0, 50.0, 95.0))
    q_rg = np.percentile(v_rg, (5.0, 50.0, 95.0))
    span_in = (q_in[2] - q_in[0]) / (q_in[1] + 1e-3)
    span_rg = (q_rg[2] - q_rg[0]) / (q_rg[1] + 1e-3)

    lap = cv2.Laplacian(cv2.cvtColor(ctx, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    tex = float(lap[oy1:oy2, ox1:ox2].var()
                / max(float(lap.reshape(-1)[ridx].var()), 1e-6))

    h_in, s_in = hh[oy1:oy2, ox1:ox2], ss[oy1:oy2, ox1:ox2]
    dh = np.abs(h_in - np.median(h_rg))
    dh = np.minimum(dh, 180.0 - dh)          # hue is circular, 0..179

    return {
        "chrom_p90": float(np.percentile(d, 90)),
        "v_ratio": float(span_in / (span_rg + 1e-3)),
        "tex": tex,
        "hue_dist": float(np.percentile(dh, 75) / 90.0),
        "sat_ratio": float(np.median(s_in) / (np.median(s_rg) + 1e-3)),
        "green_frac": float(((h_in >= 30) & (h_in <= 90) & (s_in > 60)).mean()),
        # q_rg[1] IS np.median(v_rg); it was being computed a second
        # time here, a whole extra partition of the ring for a number
        # already in hand four lines up.
        "dark_flat": float((v_in < q_rg[1] * 0.75).mean()),
    }


def person_score_fast(frame, box):
    """
    Probability the box is a real object rather than shadow or
    vegetation. None is an ABSTENTION - the caller must keep the box
    and its confidence, never drop it. See robust_detect.person_score.
    """
    f = shadow_features_fast(frame, box)
    if f is None:
        return None
    v = np.array([f[k] for k in SHADOW_FEATURES], np.float32)
    v = np.clip(np.nan_to_num(v), -1e4, 1e4)
    z = float(((v - SHADOW_MEAN) / SHADOW_SCALE * SHADOW_COEF).sum()
              + SHADOW_BIAS)
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))


# =========================================================
# DETECTOR
# =========================================================

class RobustPersonDetector:
    """
    Multi-scale tiled YOLOE with weighted box fusion and a photometric
    shadow gate, with the whole per-frame path kept on the GPU and,
    where CUDA allows it, inside a single captured graph.

    Same constructor keywords as robust_detect.RobustPersonDetector,
    plus precision / backend / max_det / rect / workers.
    """

    def __init__(self,
                 weights=WEIGHTS,
                 device=None,
                 imgsz=IMGSZ,
                 conf_floor=CONF_FLOOR,
                 positives=None,
                 negatives=None,
                 tiling=ENABLE_TILING,
                 tile_scale=TILE_TARGET_SCALE,
                 tile_overlap=TILE_OVERLAP,
                 tile_max=TILE_MAX,
                 shadow_reject=SHADOW_REJECT,
                 shadow_conf_weight=SHADOW_CONF_WEIGHT,
                 precision=DEFAULT_PRECISION,
                 backend=DEFAULT_BACKEND,
                 max_det=MAX_DET,
                 rect=RECT_INPUT,
                 antialias=ANTIALIAS,
                 workers=SHADOW_WORKERS,
                 cv_threads=None,
                 gsd=None,
                 person_min_m=PERSON_MIN_M,
                 person_max_m=PERSON_MAX_M,
                 fast_physics=True,
                 verbose=True):

        self.cv_threads = cv_threads
        # Ground sample distance, metres per pixel. None disables the
        # size gate entirely (the reference abstains the same way), so
        # a caller that has no telemetry gets exactly the old
        # behaviour. See detect() for the per-frame override.
        self.gsd = gsd
        self.person_min_m = person_min_m
        self.person_max_m = person_max_m
        self.device = pick_device(device)
        if self.device.type == "cuda":
            _tune_torch()
            _tune_opencv(self.cv_threads)

        # fp16 is a GPU story. On CPU it is slower than fp32 and on
        # some backends unimplemented, so do not let a preset force it.
        if self.device.type == "cpu":
            precision = "fp32"
        if precision not in _DTYPES:
            raise ValueError(f"precision must be one of {sorted(_DTYPES)}")
        self.precision = precision
        self.dtype = _DTYPES[precision]

        if backend not in ("graph", "eager", "compile"):
            raise ValueError("backend must be graph, eager or compile")
        # Graph capture is a CUDA feature. Anywhere else, run eager.
        self.backend = backend if self.device.type == "cuda" else "eager"

        self.imgsz = imgsz
        self.conf_floor = conf_floor
        self.tiling = tiling
        self.tile_scale = tile_scale
        self.tile_overlap = tile_overlap
        self.tile_max = tile_max
        self.shadow_reject = shadow_reject
        self.shadow_conf_weight = shadow_conf_weight
        self.max_det = int(max_det)
        self.rect = rect
        self.antialias = antialias
        self.fast_physics = fast_physics
        self.verbose = verbose

        self.positives = list(positives if positives is not None
                              else POSITIVE_PROMPTS)
        self.negatives = list(negatives if negatives is not None
                              else NEGATIVE_PROMPTS)
        self.prompts = self.positives + self.negatives
        self.n_pos = len(self.positives)
        self._prompt_arr = np.array(self.prompts, dtype=object)

        self._build_model(weights)

        self._pool = ThreadPoolExecutor(
            max_workers=max(1, workers or 4),
            thread_name_prefix="shadow") if workers != 1 else None

        self._plans = {}
        self._state = {}          # per-resolution GPU buffers and graph
        self._graph_note = None
        self.stats = {"frames": 0, "raw": 0, "seam": 0,
                      "fused": 0, "shadow": 0, "geom": 0, "size": 0,
                      "out": 0}
        self.timing = {"upload": 0.0, "infer": 0.0, "download": 0.0,
                       "decode": 0.0, "fuse": 0.0, "physics": 0.0}

        if verbose:
            print(f"[cuda] weights={weights} device={self.device} "
                  f"imgsz={imgsz} precision={precision} backend={self.backend}")
            print(f"[cuda] prompts: {self.n_pos} positive "
                  f"+ {len(self.negatives)} negative | max_det={self.max_det}")
            if tiling:
                p = build_plan(1920, 1080, imgsz, tiling, tile_scale,
                               tile_overlap, tile_max, rect)
                print(f"[cuda] tiling on: {len(p.tiles)} tiles at 1080p "
                      f"(scale {tile_scale}, overlap {tile_overlap}) "
                      f"-> batch {p.batch} at {p.canvas_w}x{p.canvas_h}")
            else:
                print("[cuda] tiling off")
            print(f"[cuda] shadow_reject={shadow_reject}")

    # ---------------- model ----------------

    def _build_model(self, weights):
        """
        Load YOLOE, set the prompt classes, then strip the model down
        to the part this pipeline actually reads.
        """
        self.yoloe = YOLOE(weights)
        self.yoloe.set_classes(self.prompts, self.yoloe.get_text_pe(self.prompts))

        net = self.yoloe.model

        # --- fix 0: pin the end-to-end head, BEFORE fuse ------------
        # _decode below reads (batch, max_det, 6) -- the NMS-free
        # output that Detect.forward only produces when end2end is on.
        # Older ultralytics defaulted that property to True whenever a
        # one2one branch existed; 8.4.144 flipped the default to False
        # unless _end2end is set explicitly. Two things then went
        # wrong, and only the first is obvious:
        #
        #   1. forward() skipped postprocess() and returned the raw
        #      one2many tensor (batch, 4+nc+nm, anchors). Slicing
        #      [..., :6] off that gives negative-width boxes and
        #      confidences above 1, every fusion cluster comes out
        #      empty, and weighted_box_fusion dies on an empty argmax.
        #
        #   2. Detect.fuse() reads end2end to decide which branch to
        #      throw away, so fusing first DELETES one2one_cv2/cv3 and
        #      the flag can no longer be turned back on -- ultralytics
        #      just warns "this model has no one-to-one head".
        #
        # Hence before the fuse() call, not after. This is not a
        # behaviour change: it restores exactly what the older
        # ultralytics did by default on this same checkpoint.
        head_pre = net.model[-1]
        if hasattr(type(head_pre), "end2end"):
            head_pre.end2end = True

        try:
            net = net.fuse(verbose=False)       # conv+bn folded
        except Exception as exc:                # noqa: BLE001 - informational
            if self.verbose:
                print(f"[cuda] fuse skipped: {exc}")
        net = net.to(self.device).eval()
        for p in net.parameters():
            p.requires_grad_(False)

        head = net.model[-1]
        self.head = head
        head.max_det = self.max_det

        # --- fix 1: never build the mask prototypes ---------------
        # YOLOESegment26.forward returns ((detect_out, proto), preds)
        # and this pipeline only ever reads detect_out. Bypassing the
        # Proto branch is bit-identical on boxes and confidences and
        # is worth ~45% of the forward. Bound on the instance only.
        if isinstance(head, YOLOEDetect):
            def _detect_only(x, _h=head):
                out = YOLOEDetect.forward(_h, x)
                return out[0] if isinstance(out, (tuple, list)) else out
            head.forward = _detect_only
            self._proto_skipped = True
        else:
            self._proto_skipped = False

        # --- fix 5a: make the head's top-k capturable --------------
        # Stock get_topk_index indexes a CUDA tensor with a CPU
        # arange, which is a pageable host-to-device copy and cannot
        # be recorded into a CUDA graph. Same maths, device-resident.
        #
        # ultralytics 8.4.144 rewrote that function around _gather and
        # _grouped_topk: it is already device-resident (so the patch
        # below buys nothing) and it returns the anchor index as
        # (batch, k) rather than (batch, k, 1), which is the shape the
        # new postprocess feeds straight into _gather. Overriding it
        # there hands postprocess a 4-D index and the whole forward
        # dies in expand(). Presence of _gather is the version test:
        # if the head has it, stock is both correct and capturable.
        if not getattr(head, "agnostic_nms", False) \
                and not hasattr(head, "_gather"):
            def _graph_safe_topk(scores, max_det, _h=head):
                b, anchors, nc = scores.shape
                k = min(max_det, anchors)
                ori = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
                s = scores.gather(dim=1, index=ori.repeat(1, 1, nc))
                s, index = s.flatten(1).topk(k)
                ar = torch.arange(b, device=scores.device)[..., None]
                idx = ori[ar, index // nc]
                return s[..., None], (index % nc)[..., None].float(), idx
            head.get_topk_index = _graph_safe_topk

        if self.dtype != torch.float32:
            net = net.to(self.dtype)
        if self.device.type in ("cuda", "xpu"):
            net = net.to(memory_format=torch.channels_last)

        # --- fix 5b: pin the prompt embedding to the device ---------
        # set_classes() stores the text embedding as net.pe, and it is
        # a plain attribute rather than a buffer, so net.to() above
        # leaves it fp32 on the CPU. YOLOEModel.predict then runs
        #
        #     get_cls_pe(...).to(device=x.device, dtype=x.dtype)
        #
        # on EVERY forward: a torch.cat plus a pageable host-to-device
        # copy of the embedding, per frame, forever. Outside a graph
        # that is merely waste; inside one it is fatal, because a
        # pageable H2D copy is exactly the operation CUDA refuses to
        # record (cudaErrorStreamCaptureUnsupported). This was the
        # capture failure -- everything else in the head was already
        # graph-safe.
        #
        # Converting it once makes the .to() a no-op that returns the
        # same tensor, and overriding get_cls_pe drops the per-forward
        # cat as well. The prompts are fixed for the detector's life,
        # so there is nothing left to recompute.
        pe = getattr(net, "pe", None)
        if pe is not None:
            pe = pe.to(device=self.device, dtype=self.dtype).contiguous()
            net.pe = pe
            # Signature matches YOLOEModel.get_cls_pe(tpe, vpe); both
            # are None here (no text prompt tensor is passed in and no
            # visual prompt exists), which is what makes the cached
            # answer correct rather than merely convenient.
            def _cached_cls_pe(tpe, vpe, _pe=pe):
                if tpe is None and vpe is None:
                    return _pe
                return type(net).get_cls_pe(net, tpe, vpe)
            net.get_cls_pe = _cached_cls_pe
            self._pe_pinned = True
        else:
            self._pe_pinned = False

        if self.backend == "compile":
            try:
                net = torch.compile(net, mode="max-autotune-no-cudagraphs",
                                    dynamic=False)
                if self.verbose:
                    print("[cuda] torch.compile enabled "
                          "(first frame will be slow)")
            except Exception as exc:            # noqa: BLE001
                print(f"[cuda] torch.compile unavailable, staying eager: {exc}")
        self.net = net

    # ---------------- per-resolution GPU state ----------------

    def _plan_for(self, w, h):
        key = (w, h)
        if key not in self._plans:
            self._plans[key] = build_plan(
                w, h, self.imgsz, self.tiling, self.tile_scale,
                self.tile_overlap, self.tile_max, self.rect)
        return self._plans[key]

    def _state_for(self, plan):
        """
        Allocate the static buffers for one resolution and, on CUDA,
        try to capture preprocessing + forward into a graph.

        Everything the graph touches has to live at a fixed address:
        the uint8 frame it reads, the padded canvas it writes through,
        and the output it leaves behind. Those three are allocated
        here and reused for the life of the detector.
        """
        key = (plan.w, plan.h)
        if key in self._state:
            return self._state[key]

        dev = self.device
        st = {"plan": plan, "graph": None}

        # Static frame buffer, plus a pinned host staging buffer. On
        # unified memory a pinned copy is markedly cheaper than a
        # pageable one and, unlike a pageable copy, it can be async.
        st["gpu_frame"] = torch.empty((plan.h, plan.w, 3),
                                      dtype=torch.uint8, device=dev)
        st["pin"] = (torch.empty((plan.h, plan.w, 3), dtype=torch.uint8,
                                 pin_memory=True)
                     if dev.type == "cuda" else None)

        # The canvas is filled with letterbox grey ONCE. Only the
        # image region is ever written after that, so the padding
        # stays correct across every graph replay without being
        # rewritten each frame.
        canvas = torch.full((plan.batch, 3, plan.canvas_h, plan.canvas_w),
                            PAD_VALUE, dtype=self.dtype, device=dev)
        if dev.type in ("cuda", "xpu"):
            canvas = canvas.contiguous(memory_format=torch.channels_last)
        st["canvas"] = canvas

        _sync(dev)
        if self.backend != "eager" and dev.type == "cuda":
            self._try_capture(st)
        if st["graph"] is None:
            st["out"] = None
        self._state[key] = st
        return st

    def _preprocess(self, st):
        """
        Frame on the GPU -> the network's input batch, without the
        host touching a pixel.

        The frame is a (h, w, 3) uint8 tensor already in device
        memory. permute gives a (3, h, w) strided VIEW, so every tile
        crop below is free - just an offset and a stride. The only
        real work is one stack (which materialises the tiles) and two
        batched resizes.
        """
        plan, canvas = st["plan"], st["canvas"]
        chw = st["gpu_frame"].permute(2, 0, 1)              # view, BGR

        lp = plan.full
        full = chw.unsqueeze(0).to(self.dtype)
        full = F.interpolate(full, size=(lp.new_h, lp.new_w),
                             mode="bilinear", align_corners=False,
                             antialias=self.antialias)
        # BGR -> RGB after the resize, on the small tensor, and scale
        # to 0..1 in the same pass.
        canvas[0:1, :, lp.pad_y:lp.pad_y + lp.new_h,
               lp.pad_x:lp.pad_x + lp.new_w] = full.flip(1).div_(255.0)

        if plan.tiles:
            tp = plan.tile
            crops = torch.stack([chw[:, y1:y2, x1:x2]
                                 for x1, y1, x2, y2 in plan.tiles])
            crops = F.interpolate(crops.to(self.dtype),
                                  size=(tp.new_h, tp.new_w),
                                  mode="bilinear", align_corners=False,
                                  antialias=self.antialias)
            canvas[1:, :, tp.pad_y:tp.pad_y + tp.new_h,
                   tp.pad_x:tp.pad_x + tp.new_w] = crops.flip(1).div_(255.0)
        return canvas

    @staticmethod
    def _head_out(y):
        """Unwrap whatever nesting the head returned down to the tensor."""
        while isinstance(y, (tuple, list)):
            y = y[0]
        return y

    def _forward(self, st):
        """Preprocess + forward, returning (batch, max_det, 6) float32."""
        x = self._preprocess(st)
        y = self.net(x)
        return self._head_out(y)[..., :6].float().contiguous()

    def _try_capture(self, st):
        """
        Record preprocessing + forward into a CUDA graph.

        This is the fix for a pinned CPU core. Without it the host
        issues one launch per kernel, every frame, forever; with it
        the host issues one replay and the GPU walks the recorded
        sequence itself. Failure here is not fatal - the detector
        says so once and runs eager.
        """
        # cudnn.benchmark and graph capture are a bad pair. _tune_torch
        # turns autotuning on because every shape here is fixed, which
        # is the textbook case for it -- but the warm-up forwards below
        # then run the exhaustive algorithm search for every conv in
        # the network, and doing that under capture conditions is
        # pathologically slow. Measured on this model, 40 frames:
        #
        #   benchmark=True    first frame 17114 ms, steady 37.0 ms
        #   benchmark=False   first frame   144 ms, steady 36.7 ms
        #
        # Seventeen seconds of startup for a steady state that is, if
        # anything, marginally worse -- cuDNN's heuristics already
        # pick well for fp16 channels_last. So autotuning is off for
        # the capture and restored afterwards, because the setting is
        # global and a caller's other models (a ReID backbone, say)
        # may legitimately want it.
        prev_benchmark = torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.benchmark = False
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):              # warmup is required
                    self._forward(st)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                out = self._forward(st)
            st["graph"] = graph
            st["out"] = out
            st["out_cpu"] = torch.empty(out.shape, dtype=out.dtype,
                                        device="cpu", pin_memory=True)
            if self.verbose and self._graph_note != "ok":
                self._graph_note = "ok"
                print(f"[cuda] graph captured: batch {out.shape[0]} "
                      f"at {st['plan'].canvas_w}x{st['plan'].canvas_h}, "
                      f"{out.shape[1]} rows/image")
        except Exception as exc:                # noqa: BLE001
            st["graph"] = None
            if self._graph_note != "failed":
                self._graph_note = "failed"
                print(f"[cuda] CUDA graph capture failed, using eager "
                      f"({type(exc).__name__}: {exc})")
        finally:
            torch.backends.cudnn.benchmark = prev_benchmark

    # ---------------- inference ----------------

    def _run_network(self, frame, st):
        """One frame in, a (batch, max_det, 6) numpy array out."""
        t0 = time.perf_counter()
        host = torch.from_numpy(frame)
        if st["pin"] is not None:
            st["pin"].copy_(host)
            st["gpu_frame"].copy_(st["pin"], non_blocking=True)
        else:
            st["gpu_frame"].copy_(host)
        t1 = time.perf_counter()

        if st["graph"] is not None:
            st["graph"].replay()
            out = st["out"]
            st["out_cpu"].copy_(out, non_blocking=True)
            torch.cuda.synchronize()
            raw = st["out_cpu"].numpy()
        else:
            with torch.inference_mode():
                out = self._forward(st)
            raw = out.to("cpu", non_blocking=False).numpy()
        t2 = time.perf_counter()

        self.timing["upload"] += t1 - t0
        self.timing["infer"] += t2 - t1
        return raw

    # ---------------- decode ----------------

    def _decode(self, raw, plan):
        """
        (batch, max_det, 6) of [x1, y1, x2, y2, conf, cls] in canvas
        pixels -> frame-space boxes, confidences and prompt ids.

        Same two rules as the original, applied to whole arrays rather
        than one box at a time: drop anything whose winning prompt is
        a negative, and drop any tile box flush against a cut edge
        because the overlapping neighbour sees that target whole.
        """
        conf = raw[..., 4]
        cls = raw[..., 5].astype(np.int32)
        alive = conf >= self.conf_floor
        self.stats["raw"] += int(alive.sum())
        keep = alive & (cls < self.n_pos)

        boxes, confs, ids = [], [], []

        # --- full-frame pass: it sees everything, no seam logic ---
        m0 = keep[0]
        if m0.any():
            lp = plan.full
            boxes.append((raw[0][m0][:, :4] - lp.offset) / lp.scale)
            confs.append(conf[0][m0])
            ids.append(cls[0][m0])

        # --- tiles: to frame coords, reject seam fragments ---
        if plan.tiles:
            kt = keep[1:]
            ti, ri = np.nonzero(kt)
            if ti.size:
                tp = plan.tile
                b = (raw[1:][ti, ri, :4] - tp.offset) / tp.scale   # tile-local
                edge = plan.interior[ti]
                clipped = (
                    ((b[:, 0] <= SEAM_MARGIN) & edge[:, 0]) |
                    ((b[:, 1] <= SEAM_MARGIN) & edge[:, 1]) |
                    ((b[:, 2] >= plan.tile_w - SEAM_MARGIN) & edge[:, 2]) |
                    ((b[:, 3] >= plan.tile_h - SEAM_MARGIN) & edge[:, 3])
                )
                self.stats["seam"] += int(clipped.sum())
                ok = ~clipped
                if ok.any():
                    org = plan.origins[ti[ok]]
                    boxes.append(b[ok] + np.concatenate([org, org], 1))
                    confs.append(conf[1:][ti, ri][ok])
                    ids.append(cls[1:][ti, ri][ok])

        if not boxes:
            return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), [])
        return (np.concatenate(boxes).astype(np.float32),
                np.concatenate(confs).astype(np.float32),
                list(self._prompt_arr[np.concatenate(ids)]))

    # ---------------- physics ----------------

    def _score_boxes(self, frame, boxes):
        """Photometric gate over every surviving box, in parallel."""
        fn = person_score_fast if self.fast_physics else person_score_reference
        if self._pool is None or len(boxes) < 2:
            return [fn(frame, b) for b in boxes]
        return list(self._pool.map(lambda b: fn(frame, b), boxes))

    # ---------------- main entry ----------------

    def detect(self, frame, gsd=None):
        """
        BGR frame -> list[Detection], sorted by confidence.

        `gsd` is metres per pixel for THIS frame, overriding the
        constructor default. It is a per-frame argument because
        altitude and zoom are per-frame: on a survey clip the ground
        sample distance moves with every metre the aircraft climbs, so
        a value fixed at construction would be wrong everywhere except
        the altitude it was measured at. Pass

            alt_rel_m / (fx * zoom_ratio)

        from telemetry. None on both means the gate abstains and the
        detector behaves exactly as it did before.
        """
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        h, w = frame.shape[:2]
        self.stats["frames"] += 1

        plan = self._plan_for(w, h)
        st = self._state_for(plan)

        raw = self._run_network(frame, st)

        t0 = time.perf_counter()
        boxes, confs, prompts = self._decode(raw, plan)
        t1 = time.perf_counter()
        self.timing["decode"] += t1 - t0
        if len(boxes) == 0:
            return []

        fb, fc, fraw, fn_, flab = weighted_box_fusion(boxes, confs, prompts)
        self.stats["fused"] += len(fb)
        t2 = time.perf_counter()
        self.timing["fuse"] += t2 - t1

        # Geometry first - it is nearly free and every box it removes
        # is a box the physics does not have to pay for.
        gsd_eff = self.gsd if gsd is None else gsd
        cand, meta = [], []
        for b, c, rawc, n, lab in zip(fb, fc, fraw, fn_, flab):
            x1 = int(max(0, min(w - 1, b[0])))
            y1 = int(max(0, min(h - 1, b[1])))
            x2 = int(max(0, min(w, b[2])))
            y2 = int(max(0, min(h, b[3])))
            bw, bh = x2 - x1, y2 - y1
            if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
                self.stats["geom"] += 1
                continue
            if (bw * bh) > MAX_BOX_FRAC * w * h:
                self.stats["geom"] += 1
                continue
            if max(bh / max(bw, 1e-6), bw / max(bh, 1e-6)) > MAX_ASPECT:
                self.stats["geom"] += 1
                continue
            # Physical size. The one gate here that needs no tuning
            # and no refitting between environments: a person seen
            # from above is between person_min_m and person_max_m
            # along their long axis, and a dropped water bottle is
            # not, at any altitude.
            #
            # This runs before the photometric gate on purpose -- it
            # is a multiply and two compares, and every box it removes
            # is a box the ~4 ms/box physics never has to score.
            #
            # Measured at 20 m / 2x (gsd 6.3 mm/px), which is where
            # this gate earned its place: a prone casualty is 1.59 m,
            # a curled one 0.76 m, a person sitting by a bag 0.51 m,
            # and the litter bottle that reached GID assignment is
            # 0.27 m and 0.24 m. The reference's negative-prompt veto
            # also rejects that bottle, but it rejects the 0.51 m
            # person along with it; this does not.
            # ...but only for a box the frame fully contains. A target
            # straddling the border is cut off by it, so its box width
            # is a LOWER BOUND on the target's real extent, not a
            # measurement of it -- and measured over 900 frames here,
            # 23 of the 25 boxes this gate removed were exactly that:
            # people walking into or out of frame, conf up to 0.89,
            # whose visible sliver happened to be under 0.30 m. Gating
            # on a lower bound would quietly delete every casualty at
            # the edge of the sensor, which is the opposite of what a
            # search pattern needs. So the gate abstains there, the
            # way the photometric gate abstains when it cannot judge.
            if gsd_eff and not (x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN
                                or x2 >= w - EDGE_MARGIN
                                or y2 >= h - EDGE_MARGIN):
                keep, _why = size_verdict(float(max(bw, bh)), gsd_eff,
                                          self.person_min_m,
                                          self.person_max_m)
                if not keep:
                    self.stats["size"] += 1
                    continue

            cand.append((x1, y1, x2, y2))
            meta.append((float(c), float(rawc), int(n), lab))

        scores = self._score_boxes(frame, cand)
        t3 = time.perf_counter()
        self.timing["physics"] += t3 - t2

        dets = []
        for (x1, y1, x2, y2), (c, rawc, n, lab), ps in zip(cand, meta, scores):
            if ps is not None and ps < self.shadow_reject:
                self.stats["shadow"] += 1
                continue
            # Physics demotes doubt and nothing else. A box it merely
            # cannot judge keeps its score, so a casualty near the
            # frame edge is never quietly penalised for being there.
            adj = confidence_adjust(ps, self.shadow_conf_weight)
            dets.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                conf=float(min(c * adj, 0.999)),
                raw_conf=rawc,
                person_score=float(ps) if ps is not None else float("nan"),
                n_views=n,
                prompt=lab,
            ))

        dets.sort(key=lambda d: -d.conf)
        self.stats["out"] += len(dets)
        return dets

    # ---------------- reporting ----------------

    def report(self):
        s = self.stats
        f = max(s["frames"], 1)
        return (f"[cuda] {s['frames']} frames | raw {s['raw']} "
                f"({s['raw'] / f:.1f}/f) -> seam-drop {s['seam']} "
                f"-> fused {s['fused']} -> shadow-drop {s['shadow']} "
                f"geom-drop {s['geom']} size-drop {s['size']} "
                f"-> out {s['out']} ({s['out'] / f:.1f}/f)")

    def timing_report(self):
        f = max(self.stats["frames"], 1)
        parts = " ".join(f"{k} {v / f * 1000:6.1f}"
                         for k, v in self.timing.items())
        total = sum(self.timing.values()) / f * 1000
        return (f"[cuda] ms/frame  {parts}  | total {total:6.1f} "
                f"({1000 / max(total, 1e-6):.1f} FPS)")

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)


# =========================================================
# LEGACY ADAPTER
# =========================================================

def detect_boxes(detector, frame):
    """Same shape as the old Pipeline_yoloe.detect(): (x1,y1,x2,y2,conf)."""
    return [d.as_tuple() for d in detector.detect(frame)]


# =========================================================
# PRESETS
# =========================================================

def make_detector(preset="balanced", env=None, **kw):
    """
    The same three operating points as robust_detect.make_detector -
    the tile scale and shadow gate are unchanged, so a preset means
    the same thing in both modules.

        fast      4 tiles, batch 5
        balanced  9 tiles, batch 10, the measured quality peak
        thorough  9 tiles + a permissive gate, for a second sweep

    Anything passed as **kw overrides the preset.
    """
    presets = {
        "fast": dict(tile_scale=0.45, shadow_reject=0.25),
        "balanced": dict(tile_scale=0.67, shadow_reject=0.25),
        "thorough": dict(tile_scale=0.67, shadow_reject=0.10,
                         conf_floor=0.04),
    }
    # This module has its own detect() and its own end2end head path,
    # so the environment profiles added to robust_detect -- the urban
    # negative vocabulary, the negative-prompt veto, the mask shape
    # gates -- are NOT implemented here. It inherits only the parts
    # that live in shared code: plan_tiles, the seam rule and the
    # hardened consensus in weighted_box_fusion. Refusing loudly beats
    # returning a grass detector that was asked for an urban one.
    if env not in (None, "grass"):
        raise NotImplementedError(
            f"robust_detect_cuda has no env={env!r} profile; the urban "
            f"gates are only in robust_detect.make_detector so far. Use "
            f"robust_detect for urban footage, or port detect().")
    if preset not in presets:
        raise ValueError(f"preset must be one of {sorted(presets)}")
    cfg = dict(presets[preset])
    cfg.update(kw)
    return RobustPersonDetector(**cfg)


# =========================================================
# THREADED VIDEO READER
# =========================================================

class ThreadedVideoReader:
    """
    Decode on its own thread so H.264 never blocks the GPU.

    1080p software decode costs ~10 ms on x86 and appreciably more on
    an Arm core. Run serially that is time the Blackwell spends idle;
    run ahead in a queue it disappears behind the forward pass.
    Hardware acceleration is requested and quietly dropped if the
    OpenCV build has no backend for it.
    """

    def __init__(self, source, queue_size=4, stride=1, hw=True, start=0,
                 cv_threads=None):
        # Before the pump thread exists, not after. OpenCV spins its
        # parallel_for pool up on first use, and first use is the
        # decode below; a detector built later then finds the pool
        # already running and resizing it no longer helps. See
        # CV_THREADS -- getting this order wrong is worth 2x.
        _tune_opencv(cv_threads)
        self.cap = None
        if hw:
            try:
                self.cap = cv2.VideoCapture(
                    source, cv2.CAP_FFMPEG,
                    [int(cv2.CAP_PROP_HW_ACCELERATION),
                     int(cv2.VIDEO_ACCELERATION_ANY)])
                if not self.cap.isOpened():
                    self.cap.release()
                    self.cap = None
            except Exception:                   # noqa: BLE001
                self.cap = None
        if self.cap is None:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {source}")

        self.stride = max(1, stride)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Seek before the pump thread starts. A survey clip is mostly
        # empty ground with the casualties in one stretch of it, so a
        # demo that always begins at frame 0 spends its first minute
        # proving nothing.
        self.start = max(0, int(start))
        if self.start:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)

        self._q = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        # Count from the seek point, so a reported frame index is the
        # index in the video and can be seeked back to.
        i = self.start
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            if i % self.stride == 0:
                self._q.put((i, frame))
            i += 1
        self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item

    def close(self):
        self._stop.set()
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:                       # noqa: BLE001
            pass
        self._thread.join(timeout=1.0)
        self.cap.release()


class ThreadedVideoWriter:
    """
    Encode on its own thread, for the same reason decoding gets one.

    cv2.VideoWriter.write() on a 1920x1080 mp4v frame costs ~13 ms,
    and run inline that lands squarely on the frame budget: measured
    end to end, 24.3 FPS detecting became 18.4 FPS the moment --save
    was passed, which would have made the recording slower than the
    live system it is meant to be evidence of. Encoding is pure CPU
    and has nothing the detector needs, so it belongs behind a queue.

    The queue is bounded: if the encoder falls permanently behind,
    this blocks rather than growing until the machine swaps.
    """

    def __init__(self, path, fps, size, queue_size=16):
        self.writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open {path} for writing")
        self._q = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        while True:
            frame = self._q.get()
            if frame is None:
                return
            self.writer.write(frame)

    def write(self, frame):
        self._q.put(frame)

    def close(self):
        """Drain what is queued before closing - never truncate."""
        self._q.put(None)
        self._thread.join()
        self.writer.release()


# =========================================================
# CLI
# =========================================================

def _draw(frame, dets):
    for d in dets:
        good = d.conf >= 0.5
        colour = (0, 220, 0) if good else (0, 170, 255)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), colour, 2)
        ps = "--" if d.person_score != d.person_score else f"{d.person_score:.2f}"
        cv2.putText(frame, f"{d.conf:.2f} p{ps} x{d.n_views}",
                    (d.x1, max(12, d.y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, 1, cv2.LINE_AA)
    return frame


def _grab_frames(source, n, spacing=150, start=300):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for i in range(n):
        idx = start + i * spacing
        if total and idx >= total:
            idx = (start + i * spacing) % total
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {source}")
    return frames


def cmd_bench(args):
    """Phase timings for the fast path, and the old path for scale."""
    frames = _grab_frames(args.source, args.bench_frames,
                          args.bench_spacing, args.bench_start)
    print(f"[bench] {len(frames)} frames at "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}\n")

    det = make_detector(args.preset, weights=args.weights, device=args.device,
                        imgsz=args.imgsz, precision=args.precision,
                        backend=args.backend, max_det=args.max_det,
                        rect=not args.square, antialias=args.antialias,
                        workers=args.workers, cv_threads=args.cv_threads,
                        gsd=args.gsd)
    for f in frames[:3]:
        det.detect(f)                            # warm cudnn / autotune
    det.stats = {k: 0 for k in det.stats}
    det.timing = {k: 0.0 for k in det.timing}

    _sync(det.device)
    t = time.perf_counter()
    for f in frames:
        det.detect(f)
    _sync(det.device)
    fast_ms = (time.perf_counter() - t) / len(frames) * 1000

    print(det.report())
    print(det.timing_report())
    print(f"\n[bench] robust_detect_cuda  {fast_ms:7.1f} ms/frame "
          f"({1000 / fast_ms:5.1f} FPS)")

    if not args.no_baseline:
        try:
            import robust_detect as RD
            base = RD.make_detector(args.preset, weights=args.weights,
                                    verbose=False)
            for f in frames[:2]:
                base.detect(f)
            _sync(det.device)
            t = time.perf_counter()
            for f in frames:
                base.detect(f)
            _sync(det.device)
            slow_ms = (time.perf_counter() - t) / len(frames) * 1000
            print(f"[bench] robust_detect       {slow_ms:7.1f} ms/frame "
                  f"({1000 / slow_ms:5.1f} FPS)")
            print(f"[bench] speedup             {slow_ms / fast_ms:7.2f}x")
        except Exception as exc:                # noqa: BLE001
            print(f"[bench] baseline unavailable: {exc}")
    det.close()


def cmd_verify(args):
    """
    Does the fast path find the same things?

    Two questions, answered separately, because they fail differently:
    the physics port has to be numerically exact (it is pure numpy, so
    any difference is a bug), while the detections only have to agree
    within fp16 and resampling noise.
    """
    frames = _grab_frames(args.source, args.verify,
                          args.bench_spacing, args.bench_start)
    print(f"[verify] {len(frames)} frames at "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}")

    det = make_detector(args.preset, weights=args.weights, device=args.device,
                        imgsz=args.imgsz, precision=args.precision,
                        backend=args.backend, max_det=args.max_det,
                        rect=not args.square, workers=args.workers,
                        cv_threads=args.cv_threads, gsd=args.gsd,
                        verbose=False)
    import robust_detect as RD
    base = RD.make_detector(args.preset, weights=args.weights, verbose=False)

    # --- 1. physics port, must be exact ---
    worst = 0.0
    n_boxes = 0
    for f in frames:
        for d in base.detect(f):
            box = (d.x1, d.y1, d.x2, d.y2)
            a = RD.person_score(f, box)
            b = person_score_fast(f, box)
            if (a is None) != (b is None):
                worst = float("inf")
            elif a is not None:
                worst = max(worst, abs(a - b))
            n_boxes += 1
    print(f"[verify] person_score over {n_boxes} boxes: "
          f"max abs diff {worst:.3e}"
          f"{'  OK' if worst < 1e-6 else '  <-- MISMATCH'}")

    # --- 2. detections ---
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = ((a[2] - a[0]) * (a[3] - a[1])
              + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / max(ua, 1e-9)

    thr = args.verify_conf
    matched = n_new = n_old = 0
    dconf = []
    for f in frames:
        new = [d for d in det.detect(f) if d.conf >= thr]
        old = [d for d in base.detect(f) if d.conf >= thr]
        n_new += len(new)
        n_old += len(old)
        taken = set()
        for o in old:
            best, bi = 0.0, -1
            for i, nd in enumerate(new):
                if i in taken:
                    continue
                v = iou(o.xyxy, nd.xyxy)
                if v > best:
                    best, bi = v, i
            if best >= 0.5:
                taken.add(bi)
                matched += 1
                dconf.append(new[bi].conf - o.conf)

    cover = matched / max(n_old, 1) * 100
    print(f"[verify] detections >= {thr}: old {n_old}, new {n_new}, "
          f"IoU>=0.5 matched {matched} ({cover:.1f}% of old)")
    if dconf:
        d = np.array(dconf)
        print(f"[verify] confidence delta on matched boxes: "
              f"mean {d.mean():+.4f}  max |d| {np.abs(d).max():.4f}")
    det.close()


def cmd_run(args):
    """Stream the video through the detector, optionally writing it out."""
    reader = ThreadedVideoReader(args.source, queue_size=args.reader_queue,
                                 stride=args.stride, hw=not args.no_hw_decode,
                                 start=args.start, cv_threads=args.cv_threads)
    print(f"[run] {args.source} {reader.w}x{reader.h} "
          f"{reader.fps:.1f} fps, {reader.total} frames")

    det = make_detector(args.preset, weights=args.weights, device=args.device,
                        imgsz=args.imgsz, precision=args.precision,
                        backend=args.backend, max_det=args.max_det,
                        rect=not args.square, antialias=args.antialias,
                        workers=args.workers, cv_threads=args.cv_threads,
                        gsd=args.gsd)

    writer = None
    if args.save:
        writer = ThreadedVideoWriter(args.save, reader.fps / args.stride,
                                     (reader.w, reader.h))

    n = 0
    t0 = time.perf_counter()
    last = t0
    try:
        for idx, frame in reader:
            dets = det.detect(frame)
            n += 1
            if writer is not None or args.show:
                vis = _draw(frame, dets)
                if writer is not None:
                    writer.write(vis)
                if args.show:
                    cv2.imshow("robust_detect_cuda", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            now = time.perf_counter()
            if now - last >= 2.0:
                print(f"  frame {idx:6d}  {n / (now - t0):5.1f} FPS  "
                      f"{len(dets)} det", flush=True)
                last = now
            if args.frames and n >= args.frames:
                break
    finally:
        dt = time.perf_counter() - t0
        reader.close()
        if writer is not None:
            writer.close()          # drains the queue first
        if args.show:
            cv2.destroyAllWindows()
        print(det.report())
        print(det.timing_report())
        print(f"[run] {n} frames in {dt:.1f}s -> {n / max(dt, 1e-6):.2f} FPS")
        if args.save:
            print(f"[run] wrote {args.save}")
        det.close()


def main():
    p = argparse.ArgumentParser(
        description="CUDA fast path for robust person detection")
    p.add_argument("--source", default="test.mp4")
    p.add_argument("--weights", default=WEIGHTS)
    p.add_argument("--preset", default="balanced",
                   choices=["fast", "balanced", "thorough"])
    p.add_argument("--device", default=None,
                   help="cuda:0 / xpu:0 / cpu (default: auto, CUDA first)")
    p.add_argument("--precision", default=DEFAULT_PRECISION,
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--backend", default=DEFAULT_BACKEND,
                   choices=["graph", "eager", "compile"])
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--max-det", type=int, default=MAX_DET)
    p.add_argument("--square", action="store_true",
                   help="letterbox to imgsz x imgsz like the original")
    p.add_argument("--antialias", action="store_true",
                   help="antialiased downscale (slightly slower, sharper)")
    p.add_argument("--workers", type=int, default=SHADOW_WORKERS,
                   help="threads for the photometric gate")
    p.add_argument("--gsd", type=float, default=None,
                   help="metres per pixel, enables the physical-size gate; "
                        "alt_rel_m / (fx * zoom_ratio), e.g. 0.0063 at "
                        "20 m with 2x zoom on the FCB")
    p.add_argument("--cv-threads", type=int, default=CV_THREADS,
                   help="OpenCV parallel_for pool (0 = leave alone); "
                        "see CV_THREADS -- 16 costs ~4x on a threaded reader")

    p.add_argument("--bench", action="store_true", help="phase timings")
    p.add_argument("--bench-frames", type=int, default=20)
    # Where to sample. The defaults land wherever a clip happens to
    # start, which on a survey flight is usually empty ground -- and a
    # bench or verify run over empty ground reports "0 detections" for
    # both paths and proves nothing. Point these at a stretch that
    # actually has people in it.
    p.add_argument("--bench-start", type=int, default=300,
                   help="first frame index to sample")
    p.add_argument("--bench-spacing", type=int, default=150,
                   help="frames between samples")
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the robust_detect comparison in --bench")
    p.add_argument("--verify", type=int, default=0, metavar="N",
                   help="compare against robust_detect over N frames")
    p.add_argument("--verify-conf", type=float, default=0.30)

    p.add_argument("--save", default=None, help="write an annotated mp4")
    p.add_argument("--show", action="store_true")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--start", type=int, default=0,
                   help="seek to this frame before processing")
    p.add_argument("--reader-queue", type=int, default=4)
    p.add_argument("--no-hw-decode", action="store_true")

    args = p.parse_args()
    if args.bench:
        cmd_bench(args)
    elif args.verify:
        cmd_verify(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()

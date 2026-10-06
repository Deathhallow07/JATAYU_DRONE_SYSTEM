"""
Live metric depth heatmap for a drone video: Depth Anything 3 relative depth
anchored to 2D-LiDAR ranges. Nothing is written to disk; results are shown in
an OpenCV window as they are computed.

LiDAR CSV (drone_cam_points.csv): one row per beam, consecutive rows with the
same pose form one sweep (-30, -15, 0, 15, 30 deg).
  lat_deg, lon_deg, alt_rel_m, roll_deg, pitch_deg, yaw_deg   pose of the sweep
  target_angle_deg     nominal beam angle
  cam_angle_deg        beam angle from the camera optical axis, already in the
                       camera frame, so no extrinsics are needed
  cam_range_m          range from the camera (blank = no return)

Per frame:
  1. The rows of one sweep are merged into one sample, and every sample is
     matched to a video frame (see align_samples). A frame uses the nearest
     sample within --max-gap frames.
  2. DA3 gives relative depth d_rel(u, v).
  3. Beam i with angle phi_i and range r_i, both in the camera frame:
        z_i = r_i * cos(phi_i)                 metric depth along the optical axis
        u_i = cx + sign * fx * tan(phi_i),  v_i = cy
     Camera and LiDAR are both hard-mounted, so phi_i is fixed relative to the
     optical axis and roll / pitch do not enter z_i or the pixel. They only
     tilt the whole view relative to the ground, which DA3 already sees.
     The scan plane is the image's horizontal plane: the FCB's vertical FOV is
     only +-19 deg, so +-30 deg beams can only land across the width.
  4. `sign` (which image side positive angles land on) comes from the
     telemetry, see beam_side_from_roll.
  5. DA3 depth is correct up to one scale factor, so z = a * d_rel with
     a = median(z_i / d_rel_i) over the beams (robust to one bad return).
     --fit affine adds an offset b when the beams span enough depth.
  6. z(u, v) = a * d_rel(u, v) + b, shown as a heatmap in metres.

Window: frame | metric depth. Hover to read the depth under the cursor,
space pauses, q / Esc quits.

Usage:
  python lidar_anchored_depth.py                         # defaults below
  python lidar_anchored_depth.py --video clip.mp4 --csv drone_cam_points.csv \
      --telem clip.csv --start 3000 --stride 2
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
VIDEO = HERE.parent / "pipeline" / "1xzoom1.5ms20m" / "fcb-20260912-092817.mp4"
LIDAR_CSV = HERE / "drone_cam_points.csv"
MODEL_NAME = "depth-anything/DA3-SMALL"
PROCESS_RES = 504                           # DA3 working resolution, as in depth_4.py

CAMERA_INTRINSICS = [                                   # SONY FCB
    [1584.02798, 0.0,        953.27546],
    [0.0,        1581.52473, 518.97355],
    [0.0,        0.0,        1.0],
]
CALIB_RES = (1920, 1080)                    # resolution CAMERA_INTRINSICS were calibrated at

POSE_COLS = ["lat_deg", "lon_deg", "alt_rel_m", "roll_deg", "pitch_deg", "yaw_deg"]
# per-column tolerance used to match a sweep to a telemetry row
POSE_SCALE = np.array([1e-6, 1e-6, 0.01, 0.01, 0.01, 0.01])

# DA3 input conventions (depth_anything_3/utils/io/input_processor.py)
PATCH = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------ LiDAR samples ------------------------------

def load_samples(csv_path):
    """Merge the per-beam rows of each sweep into one sample dict."""
    df = pd.read_csv(csv_path)
    missing = [c for c in POSE_COLS + ["target_angle_deg", "cam_range_m", "cam_angle_deg"]
               if c not in df.columns]
    if missing:
        sys.exit(f"{csv_path} is missing columns: {missing}")
    new_sweep = (df[POSE_COLS].ne(df[POSE_COLS].shift()).any(axis=1)
                 | (df["target_angle_deg"].diff() <= 0))
    samples = []
    for _, g in df.groupby(new_sweep.cumsum(), sort=True):
        r = g["cam_range_m"].to_numpy(float)
        phi = np.radians(g["cam_angle_deg"].to_numpy(float))
        ok = np.isfinite(r) & (r > 0) & np.isfinite(phi)
        s = {c: float(g[c].iloc[0]) for c in POSE_COLS}
        s["phi"], s["range"] = phi[ok], r[ok]
        samples.append(s)
    return samples


def dp_align(G, F):
    """Strictly increasing row of F for every row of G, minimising total pose mismatch."""
    n, m = len(G), len(F)
    if n > m:
        sys.exit(f"more LiDAR sweeps ({n}) than telemetry rows ({m})")

    def cost(i):
        d = np.abs(F - G[i])
        d[:, 5] = np.abs((F[:, 5] - G[i, 5] + 180) % 360 - 180)  # yaw wraps
        return np.nan_to_num((d / POSE_SCALE).sum(1), nan=1e9)

    acc, back, cols = cost(0), np.zeros((n, m), np.int32), np.arange(m)
    for i in range(1, n):
        best = np.minimum.accumulate(acc)
        arg = np.maximum.accumulate(np.where(acc == best, cols, 0))
        back[i, 1:] = arg[:-1]
        acc = cost(i) + np.r_[np.inf, best[:-1]]
    idx = np.empty(n, int)
    idx[-1] = int(np.argmin(acc))
    for i in range(n - 1, 0, -1):
        idx[i - 1] = back[i, idx[i]]
    return idx


def align_samples(samples, telem, n_frames):
    """Video frame of every sample, plus per-frame zoom (or None).

    The LiDAR CSV has no timestamps, but its poses are the flight controller's,
    so with the video's telemetry CSV (one row per frame) each sweep is matched
    to the frame whose pose agrees, in order. Without it, sweeps are spread
    evenly over the video.
    """
    if telem is None:
        print("no telemetry CSV: spreading LiDAR sweeps evenly over the video")
        return np.round(np.linspace(0, n_frames - 1, len(samples))).astype(int), None
    t = pd.read_csv(telem)
    Fp = t[POSE_COLS].to_numpy(float)
    G = np.array([[s[c] for c in POSE_COLS] for s in samples])
    rows = dp_align(G, Fp)
    resid = np.abs(Fp[rows, 2:5] - G[:, 2:5]).max(1)
    frames = t["frame"].to_numpy(int) if "frame" in t.columns else np.arange(len(t))
    zoom = None
    if "zoom_ratio" in t.columns:
        zoom = np.ones(n_frames)
        z = t["zoom_ratio"].to_numpy(float)
        ok = np.isfinite(z) & (z > 0) & (frames < n_frames)
        zoom[frames[ok]] = z[ok]
    print(f"matched {len(samples)} sweeps to frames via {Path(telem).name}: median gap "
          f"{np.median(np.diff(frames[rows])):.0f} frames, pose residual median "
          f"{np.median(resid):.3f} / p99 {np.percentile(resid, 99):.3f}")
    return frames[rows], zoom


def nearest_sample(sample_frames, n_frames, max_gap):
    """Per video frame: index of the nearest sample within max_gap frames, else -1."""
    q = np.arange(n_frames)
    j = np.clip(np.searchsorted(sample_frames, q), 1, len(sample_frames) - 1)
    pick = np.where(q - sample_frames[j - 1] <= sample_frames[j] - q, j - 1, j)
    return np.where(np.abs(sample_frames[pick] - q) <= max_gap, pick, -1)


def beam_side_from_roll(samples, min_alt=5.0, min_roll_deg=2.0):
    """Image side (+1 right, -1 left) that positive cam_angle points to, from the ranges.

    The scan runs across the drone, so rolling tilts one end of the sweep down
    and the other up. Over flat ground the vertical height of every return,
        r * (cos(phi) * cos(roll) + s * sin(phi) * sin(roll)),
    must agree within a sweep, where s = +1 if positive phi points to the
    drone's right (pitch scales all beams of a sweep alike and drops out).
    Every airborne, rolled sweep with returns on both sides votes for the s
    that makes its heights agree better. Roll is the flight controller's
    (right wing down positive) and image right is body right (R_BODY_CAM in
    coordinate_transformer.py), so s is also the image side.
    Returns (s or None, share of votes for s, number of votes).
    """
    votes = []
    for smp in samples:
        phi, r = smp["phi"], smp["range"]
        if (smp["alt_rel_m"] < min_alt or abs(smp["roll_deg"]) < min_roll_deg
                or len(phi) < 3 or not (phi.min() < -0.2 and phi.max() > 0.2)):
            continue
        roll = np.radians(smp["roll_deg"])
        spread = {}
        for sg in (1, -1):
            h = r * (np.cos(phi) * np.cos(roll) + sg * np.sin(phi) * np.sin(roll))
            spread[sg] = np.std(h) / np.mean(h)
        votes.append(1 if spread[1] < spread[-1] else -1)
    if not votes:
        return None, np.nan, 0
    share_right = np.mean(np.array(votes) == 1)
    s = 1 if share_right > 0.5 else -1
    return s, max(share_right, 1 - share_right), len(votes)


def fit_depth(d_rel, z, mode, min_spread):
    """Metric scale for DA3 depth from the beams. (a, b, rmse, mode) or None.

    scale:  z = a * d_rel, a = median(z_i / d_rel_i)
    affine: z = a * d_rel + b by least squares, only when d_rel spans enough
            (min_spread, relative) for b to be determined; otherwise scale.
    """
    d, z = np.asarray(d_rel, float), np.asarray(z, float)
    ok = np.isfinite(d) & np.isfinite(z) & (d > 0) & (z > 0)
    d, z = d[ok], z[ok]
    if len(d) == 0:
        return None
    if mode == "affine" and len(d) >= 3 and np.ptp(d) > min_spread * d.mean():
        A = np.stack([d, np.ones_like(d)], 1)
        (a, b), *_ = np.linalg.lstsq(A, z, rcond=None)
        if a > 0:  # farther must stay farther
            return float(a), float(b), float(np.sqrt(np.mean((a * d + b - z) ** 2))), "affine"
    a = float(np.median(z / d))
    return a, 0.0, float(np.sqrt(np.mean((a * d - z) ** 2))), "scale"


# ------------------------------- model / GPU -------------------------------

def bootstrap_da3():
    """Import DepthAnything3, from the checkout next to this script if it is not installed.

    DA3's file-export helpers (depth_anything_3.utils.export) import pycolmap,
    moviepy, open3d, ... Nothing is exported here, so when those are missing the
    module is replaced by a stub instead of failing the whole import.
    """
    import importlib.util
    import types
    if importlib.util.find_spec("depth_anything_3") is None:
        for local in (HERE / "depth-anything-3" / "src",
                      HERE / "depth-anything-3" / "depth-anything-3" / "src"):
            if (local / "depth_anything_3").is_dir():
                sys.path.insert(0, str(local))
                break
        else:
            raise SystemExit("depth_anything_3 not found: pip install it or keep the checkout "
                             "at depth-anything-3/src next to this script")
    try:
        import depth_anything_3.utils.export  # noqa: F401
    except ImportError as e:
        print(f"[load] DA3 export helpers unavailable ({e}); not needed for live inference")
        stub = types.ModuleType("depth_anything_3.utils.export")
        stub.export = lambda *a, **k: None
        sys.modules["depth_anything_3.utils.export"] = stub
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3


def processed_size(w, h, res):
    """DA3 'upper_bound_resize': longest side -> res, each side to the nearest multiple of 14."""
    s = res / max(w, h)

    def snap(x):
        down = (x // PATCH) * PATCH
        return down + PATCH if (down + PATCH) - x <= x - down else max(down, PATCH)

    return snap(max(1, round(w * s))), snap(max(1, round(h * s)))


class FrameReader(threading.Thread):
    """Decodes frames on a CPU thread into a bounded queue of (idx, bgr)."""

    def __init__(self, cap, start, stride, max_frames, depth):
        super().__init__(daemon=True)
        self.cap, self.start_idx, self.stride, self.max_frames = cap, start, stride, max_frames
        self.q = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.error = None

    def run(self):
        try:
            if self.start_idx:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_idx)
            idx, taken = self.start_idx - 1, 0
            while not self.stop.is_set() and (not self.max_frames or taken < self.max_frames):
                if not self.cap.grab():
                    break
                idx += 1
                if (idx - self.start_idx) % self.stride:
                    continue
                ok, frame = self.cap.retrieve()
                if not ok:
                    break
                while not self.stop.is_set():
                    try:
                        self.q.put((idx, frame), timeout=0.2)
                        break
                    except queue.Full:
                        pass
                taken += 1
        except Exception as e:
            self.error = e
        finally:
            while True:
                try:
                    self.q.put(None, timeout=0.2)
                    break
                except queue.Full:
                    if self.stop.is_set():
                        break


class DepthPipeline:
    """Batched DA3 inference and LiDAR anchoring on the GPU, at display resolution."""

    def __init__(self, model, device, W, H, K, samples, lookup, zoom, args):
        self.model, self.device, self.W, self.H, self.K = model, device, W, H, K
        self.samples, self.lookup, self.zoom, self.args = samples, lookup, zoom, args
        self.pw, self.ph = processed_size(W, H, args.process_res)
        self.vh, self.vw = args.vis_height, int(round(W * args.vis_height / H))
        self.s = self.vh / H  # full-res pixel -> display pixel
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
        self.lut_np = lut
        self.lut = torch.from_numpy(lut).to(device)
        self.sign = args.sign
        self.last_fit = None
        self.range = tuple(args.depth_range) if args.depth_range else None

    def relative_depth(self, frames):
        """(B, H, W, 3) uint8 BGR on GPU -> DA3 relative depth (B, vh, vw)."""
        B = frames.shape[0]
        x = frames.permute(0, 3, 1, 2).flip(1).float().div_(255)  # BGR -> RGB, B,3,H,W
        x = F.interpolate(x, size=(self.ph, self.pw), mode="bilinear", antialias=True, align_corners=False)
        x = (x - self.mean) / self.std
        if B < self.args.batch:  # keep shapes fixed for cuDNN autotune / torch.compile
            x = torch.cat([x, x[-1:].expand(self.args.batch - B, -1, -1, -1)])
        out = self.model(x[:, None], export_feat_layers=[])  # (B, 1, 3, h, w): B independent single views
        d = out["depth"][:B].float().reshape(B, 1, self.ph, self.pw)
        return F.interpolate(d, size=(self.vh, self.vw), mode="bilinear", align_corners=False)[:, 0]

    def beams(self, sample, zoom, sign):
        """Full-res pixel (u, v) and optical-axis depth z of every beam that lands in the frame."""
        fx, _, cx, cy = self.K
        u = cx + sign * fx * zoom * np.tan(sample["phi"])
        keep = (u >= 0) & (u < self.W)
        z = sample["range"] * np.cos(sample["phi"])
        return u[keep], np.full(keep.sum(), cy), z[keep]

    def d_rel_at(self, d_rel, f_idx, us, vs):
        """Median of d_rel in a (2k+1)^2 window at each full-res pixel, one gather for all."""
        k, dev = self.args.patch, self.device
        off = torch.arange(-k, k + 1, device=dev)
        fi = torch.tensor(f_idx, device=dev).view(-1, 1, 1)
        vv = torch.tensor(np.round(np.asarray(vs) * self.s), device=dev, dtype=torch.long).view(-1, 1, 1)
        uu = torch.tensor(np.round(np.asarray(us) * self.s), device=dev, dtype=torch.long).view(-1, 1, 1)
        vv = (vv + off.view(1, -1, 1)).clamp_(0, self.vh - 1)
        uu = (uu + off.view(1, 1, -1)).clamp_(0, self.vw - 1)
        return d_rel[fi, vv, uu].flatten(1).nanmedian(dim=1).values.cpu().numpy()

    @torch.inference_mode()
    def __call__(self, items):
        """Process a batch of (idx, bgr). Returns one display dict per frame."""
        dev, a = self.device, self.args
        frames = torch.from_numpy(np.stack([it[1] for it in items])).to(dev, non_blocking=True)
        d_rel = self.relative_depth(frames)

        # LiDAR anchors for every frame that has a sweep nearby
        groups, f_idx, us, vs = [], [], [], []
        for f, (idx, _) in enumerate(items):
            j = self.lookup[idx] if idx < len(self.lookup) else -1
            if j < 0:
                continue
            zoom = self.zoom[idx] if self.zoom is not None else 1.0
            u, v, z = self.beams(self.samples[j], zoom, self.sign)
            groups.append((f, j, u, v, z, len(f_idx)))
            f_idx += [f] * len(u)
            us += list(u)
            vs += list(v)
        med = self.d_rel_at(d_rel, f_idx, us, vs) if f_idx else np.empty(0)

        per_frame = [None] * len(items)
        for f, j, u, v, z, start in groups:
            d = med[start:start + len(u)]
            ok = np.isfinite(d) & (d > 0)
            fit = fit_depth(d[ok], z[ok], a.fit, a.min_spread)
            per_frame[f] = fit and dict(fit=fit, n=int(ok.sum()), sample=j,
                                        px=list(zip(u[ok], v[ok])), z=z[ok], d=d[ok])

        params, metas = [], []
        for f, (idx, _) in enumerate(items):
            chosen = per_frame[f]
            if chosen:
                a_, b_, rmse, mode = chosen["fit"]
                self.last_fit = (a_, b_)
                resid = a_ * chosen["d"] + b_ - chosen["z"]
                info = (f"#{idx}  sweep {chosen['sample']}  {chosen['n']} beams  "
                        f"a={a_:.3g} b={b_:.2f}  rmse {rmse:.2f} m [{mode}]")
                px = chosen["px"]
                lidar = [(p, zl, r) for p, zl, r in zip(px, chosen["z"], resid)]
            elif self.last_fit is not None:
                a_, b_ = self.last_fit
                info, lidar = f"#{idx}  no LiDAR sweep within {a.max_gap} frames - previous fit", []
            else:
                a_ = b_ = None
                info, lidar = f"#{idx}  waiting for first LiDAR fit", []
            params.append((a_, b_))
            metas.append(dict(frame=idx, info=info, lidar=lidar))

        n = len(items)
        valid = [p[0] is not None for p in params]
        f32 = dict(device=dev, dtype=torch.float32)  # fits are numpy float64; keep the maps float32
        av = torch.tensor([float(p[0] or 0.0) for p in params], **f32).view(n, 1, 1)
        bv = torch.tensor([float(p[1] or 0.0) for p in params], **f32).view(n, 1, 1)
        z = d_rel * av + bv

        # colour range: fixed, or 2-98th percentile smoothed over frames to avoid flicker
        q = torch.quantile(z[:, ::4, ::4].flatten(1), torch.tensor([0.02, 0.98], **f32), dim=1).T.cpu().numpy()
        lo_hi = []
        for ok, (lo, hi) in zip(valid, q):
            if not self.args.depth_range and ok and np.isfinite(lo) and hi > lo:
                self.range = (lo, hi) if self.range is None else tuple(
                    0.85 * np.array(self.range) + 0.15 * np.array((lo, hi)))
            lo_hi.append(self.range or (0.0, 1.0))
        lo = torch.tensor([float(r[0]) for r in lo_hi], **f32).view(n, 1, 1)
        hi = torch.tensor([float(r[1]) for r in lo_hi], **f32).view(n, 1, 1)
        heat = self.colorize(z, lo, hi)

        img = F.interpolate(frames.permute(0, 3, 1, 2).float(), size=(self.vh, self.vw), mode="area")
        img = img.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        heat, z = heat.cpu().numpy(), z.cpu().numpy()
        for f, m in enumerate(metas):
            if not valid[f]:
                heat[f] = 0
            m.update(rgb=img[f], heat=heat[f], z=z[f] if valid[f] else None, range=lo_hi[f])
        return metas

    def colorize(self, z, lo, hi):
        t = ((hi - z) / (hi - lo)).nan_to_num_(0.0).clamp_(0, 1)  # near = red, far = blue
        return self.lut[(t * 255).to(torch.long)]


# --------------------------------- display ---------------------------------

BAR_W = 90  # width of the colour-bar strip right of the heatmap


def put_text(img, text, org, scale=0.6):
    """White text on a dark box (a thick outline changes the glyph advance in OpenCV)."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def compose(m, pipe, mouse, rate):
    """frame | heatmap | colour bar, with LiDAR anchors, info and the depth under the cursor."""
    vw, vh, s = pipe.vw, pipe.vh, pipe.s
    strip = np.full((vh, BAR_W, 3), 30, np.uint8)
    canvas = np.hstack([m["rgb"], m["heat"], strip])
    for (u, v), zl, r in m["lidar"]:
        for k in range(2):
            c = (int(round(u * s)) + k * vw, int(round(v * s)))
            cv2.circle(canvas, c, 6, (255, 255, 255), -1)
            cv2.circle(canvas, c, 6, (0, 0, 0), 1)
        put_text(canvas, f"{zl:.1f}m", (c[0] - 20, c[1] - 14), 0.45)

    # colour bar: top = far, bottom = near
    lo, hi = m["range"]
    x0, y0, bh = 2 * vw + 12, 50, vh - 100
    bar = pipe.lut_np[np.linspace(0, 255, bh).astype(np.uint8)]
    canvas[y0:y0 + bh, x0:x0 + 16] = bar[:, None, :]
    cv2.rectangle(canvas, (x0, y0), (x0 + 16, y0 + bh), (255, 255, 255), 1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        put_text(canvas, f"{hi - frac * (hi - lo):.1f}", (x0 + 24, y0 + int(frac * bh) + 5), 0.45)
    put_text(canvas, "m", (x0 + 2, y0 - 16), 0.5)

    put_text(canvas, m["info"], (10, 26))
    put_text(canvas, f"metric depth  {rate:.1f} fps", (vw + 10, 26))

    if mouse["xy"] is not None and m["z"] is not None:
        x, y = mouse["xy"]
        if 0 <= x < 2 * vw and 0 <= y < vh:
            x %= vw
            d = float(m["z"][y, x])
            for k in range(2):
                cv2.drawMarker(canvas, (x + k * vw, y), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            put_text(canvas, f"{d:.2f} m", (x + vw + 12, y - 10), 0.7)
    return canvas


# ----------------------------------- main -----------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("--csv", default=str(LIDAR_CSV), help="LiDAR sweeps (drone_cam_points.csv)")
    ap.add_argument("--telem", default="auto",
                    help="per-frame telemetry CSV of the video, used to place the sweeps in time; "
                         "'auto' = <video>.csv if it exists, 'none' = spread sweeps evenly")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--process-res", type=int, default=PROCESS_RES, help="DA3 inference resolution")
    ap.add_argument("--batch", type=int, default=2, help="frames per GPU forward pass")
    ap.add_argument("--compile", action="store_true", help="torch.compile the DA3 backbone and head")
    ap.add_argument("--beam-sign", choices=["auto", "1", "-1"], default="auto",
                    help="1: positive cam_angle lands right of the principal point, -1: left; "
                         "auto: from how the ranges react to roll (beam_side_from_roll)")
    ap.add_argument("--fit", choices=["scale", "affine"], default="scale",
                    help="DA3 -> metres: scale only (default, DA3 depth has no offset) or scale + offset")
    ap.add_argument("--max-gap", type=int, default=6, help="max frames between a frame and its LiDAR sweep")
    ap.add_argument("--min-spread", type=float, default=0.05,
                    help="--fit affine: relative d_rel spread over the beams needed to fit an offset b")
    ap.add_argument("--patch", type=int, default=2, help="half-size of the d_rel window at a beam (display px)")
    ap.add_argument("--start", type=int, default=0, help="first video frame")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (0 = all)")
    ap.add_argument("--depth-range", type=float, nargs=2, metavar=("NEAR", "FAR"),
                    help="fixed colour range in metres (default: auto per frame, smoothed)")
    ap.add_argument("--vis-height", type=int, default=540)
    return ap.parse_args()


def main():
    args = parse_args()
    DepthAnything3 = bootstrap_da3()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # depth_anything_3.api turns autotune off on import; input shapes here are fixed, so turn it on
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")  # TF32 for DA3's fp32 depth head

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open video {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Km = np.asarray(CAMERA_INTRINSICS, float)
    sx, sy = W / CALIB_RES[0], H / CALIB_RES[1]
    K = (Km[0, 0] * sx, Km[1, 1] * sy, Km[0, 2] * sx, Km[1, 2] * sy)

    samples = load_samples(args.csv)
    telem = None if args.telem == "none" else args.telem
    if telem == "auto":
        telem = str(Path(args.video).with_suffix(".csv"))
        telem = telem if Path(telem).exists() else None
    sample_frames, zoom = align_samples(samples, telem, n_video)
    lookup = nearest_sample(sample_frames, n_video, args.max_gap)
    if args.beam_sign == "auto":
        args.sign, share, n_votes = beam_side_from_roll(samples)
        if args.sign is None:
            args.sign = -1
            print("beam side: no rolled sweeps to check against, assuming positive "
                  "cam_angle -> image left (--beam-sign -1)")
        else:
            print(f"beam side from roll: positive cam_angle -> image "
                  f"{'right' if args.sign > 0 else 'left'} ({share:.0%} of {n_votes} sweeps agree)")
    else:
        args.sign = int(args.beam_sign)
    n_beams = sum(len(s["range"]) for s in samples)
    print(f"video {W}x{H} @ {fps:.2f} fps, {n_video} frames | {len(samples)} LiDAR sweeps, "
          f"{n_beams} returns | {np.mean(lookup >= 0):.0%} of frames within {args.max_gap} frames of a sweep")

    print(f"[load] {args.model} -> {device}")
    model = DepthAnything3.from_pretrained(args.model).to(device).eval()
    if args.compile:
        net = model.model
        net.backbone = torch.compile(net.backbone, dynamic=False)
        net.head = torch.compile(net.head, dynamic=False)

    pipe = DepthPipeline(model, device, W, H, K, samples, lookup, zoom, args)
    print(f"DA3 input {pipe.pw}x{pipe.ph}, batch {args.batch}. Space pauses, q / Esc quits.")

    win = "LiDAR-anchored depth"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 2 * pipe.vw + BAR_W, pipe.vh)
    mouse = {"xy": None}
    cv2.setMouseCallback(win, lambda ev, x, y, *_: mouse.update(xy=(x, y)))

    reader = FrameReader(cap, args.start, args.stride, args.max_frames, depth=2 * args.batch)
    reader.start()
    rate, done, quit_ = 0.0, False, False
    try:
        while not done and not quit_:
            batch = []
            while len(batch) < args.batch:
                item = reader.q.get()
                if item is None:
                    done = True
                    break
                batch.append(item)
            if not batch:
                break
            t0 = time.perf_counter()
            metas = pipe(batch)
            dt = (time.perf_counter() - t0) / len(batch)
            rate = 1.0 / dt if rate == 0 else 0.9 * rate + 0.1 / dt
            for m in metas:
                paused = False
                while True:
                    cv2.imshow(win, compose(m, pipe, mouse, rate))
                    key = cv2.waitKey(30 if paused else 1) & 0xFF
                    if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        quit_ = True
                        break
                    if key == ord(" "):
                        paused = not paused
                    if not paused:
                        break
                if quit_:
                    break
    finally:
        reader.stop.set()
        while reader.is_alive():  # drain so a blocked put() can see the stop flag
            try:
                reader.q.get(timeout=0.1)
            except queue.Empty:
                pass
        reader.join()
        cap.release()
        cv2.destroyAllWindows()
    if reader.error:
        raise RuntimeError("reader thread failed") from reader.error


if __name__ == "__main__":
    main()

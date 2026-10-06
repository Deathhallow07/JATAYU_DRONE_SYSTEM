import csv
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

# --------------------------- CONFIG ---------------------------
VIDEO_PATH  = "/home/uas/localisation/yoloe/videos/fcb_tmux_bags/1xzoom1.5ms20m/fcb-20260912-092817.mp4"  # or 0 for webcam
MODEL_NAME  = "depth-anything/DA3-SMALL"  # HF repo id (cached in ~/.cache/huggingface)

START_FRAME = 5000          # Frame index to start the video from (ignored for webcam)
PROCESS_RES = 504        # Trained working resolution for DA3
FRAME_STRIDE = 1         # Process every Nth frame
MAX_DISPLAY_W = 1920     # Max display width
UPSAMPLE = cv2.INTER_LINEAR  # Linear upsampling prevents boundary ringing

# --- Rendering ---
COLORMAP    = cv2.COLORMAP_INFERNO
VIEW_MODE   = "side"     # "side" = RGB | depth · "depth" = depth only · "overlay" = blended
BLEND_ALPHA = 0.6        
COLOR_ON_DISPARITY = True    # Colour 1/depth
NORM_MODE   = "linear"   # CHANGED FROM "rank": Linear preserves flat-ground depth scale
DISP_PCT    = (0.5, 99.5) # Percentile clipping range
DISPLAY_GAMMA = 1.0      
CLAHE_CLIP  = 2.5        
CLAHE_GRID  = 8

# Ground Plane Relative Normalization
PLANE_ITERS = 80         
PLANE_INLIER_FRAC = 0.01 
PLANE_SPAN_PCT = 98.0    
PLANE_COLORMAP = cv2.COLORMAP_TURBO   
CMAP_LEVELS = 4096       
SPATIAL_SMOOTH = False   
BILATERAL_D, BILATERAL_COLOR, BILATERAL_SPACE = 5, 0.03, 7

# --- Masking ---
CONF_PERCENTILE = 15.0   
USE_CONF_MASK   = True
USE_SKY_MASK    = True   
DIM_INVALID     = False  

# --- Camera Calibration ---
INTRINSICS = (393.12779563, 394.76440916, 321.48263787, 241.58044155)
INTRINSICS_RES = None

# Extrinsics CSV
EXTRINSICS_CSV = "frame_extrinsics.csv"
EXTRINSICS_CONVENTION = "w2c"  
QUAT_WXYZ      = False   
EULER_RADIANS  = False   
EULER_ORDER    = "zyx"   
CAM_FROM_BODY = None
POSE_MAX_FRAME_GAP = 2   
POSE_FRAME_OFFSET  = -1  

# Set to 1 for robust single-frame depth; set > 1 only with accurate multi-view poses
WINDOW_SIZE   = 1        
WINDOW_STRIDE = 5        

# --- Robustness & Output ---
MAX_READ_FAILURES = 30   
OOM_RETRY = True         
OUT_DIR   = Path("da3_out")
VIDEO_OUT = OUT_DIR / "depth.mp4"
VIDEO_FPS = 20.0
# --------------------------------------------------------------


# --------------------------- CALIBRATION ---------------------------

def _quat_to_R(q, wxyz=False):
    q = np.asarray(q, np.float64)
    w, x, y, z = (q[0], q[1], q[2], q[3]) if wxyz else (q[3], q[0], q[1], q[2])
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _euler_to_R(rx, ry, rz, radians=False, order="zyx"):
    if not radians:
        rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry),
                              np.sin(ry), np.cos(rz), np.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    mats = {"x": Rx, "y": Ry, "z": Rz}
    R = np.eye(3)
    for axis in order:
        R = R @ mats[axis]
    return R


def _invert_rt(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def load_intrinsics(spec, frame_wh, calib_wh=None):
    if spec is None:
        return None

    if isinstance(spec, (str, Path)):
        p = Path(spec)
        if not p.exists():
            print(f"[calib] WARNING: INTRINSICS file not found: {p}. Proceeding without intrinsics.")
            return None
        if p.suffix == ".npy":
            K = np.load(p)
        elif p.suffix == ".json":
            import json
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                K = (np.asarray(data["K"], np.float64) if "K" in data else
                     np.array([[data["fx"], 0, data["cx"]],
                               [0, data["fy"], data["cy"]], [0, 0, 1]], np.float64))
            else:
                K = np.asarray(data, np.float64)
        else:
            K = np.loadtxt(p, delimiter="," if p.suffix == ".csv" else None)
    else:
        K = np.asarray(spec, np.float64)

    K = np.asarray(K, np.float64).squeeze()
    if K.size == 4 and K.ndim == 1:
        fx, fy, cx, cy = K
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    if K.shape != (3, 3):
        print(f"[calib] WARNING: INTRINSICS must resolve to 3x3 matrix, got {K.shape}. Ignoring.")
        return None

    if calib_wh is not None:
        sx, sy = frame_wh[0] / float(calib_wh[0]), frame_wh[1] / float(calib_wh[1])
        K = K.copy()
        K[0] *= sx
        K[1] *= sy
    return K.astype(np.float32)


def _pose_row_to_matrix(vals):
    n = len(vals)
    T = np.eye(4)
    if n >= 16:
        return np.asarray(vals[:16], np.float64).reshape(4, 4)
    if n >= 12:
        T[:3, :4] = np.asarray(vals[:12], np.float64).reshape(3, 4)
        return T
    if n >= 7:
        T[:3, :3] = _quat_to_R(vals[3:7], QUAT_WXYZ)
        T[:3, 3] = vals[0:3]
        return T
    if n >= 6:
        T[:3, :3] = _euler_to_R(*vals[3:6], radians=EULER_RADIANS, order=EULER_ORDER)
        T[:3, 3] = vals[0:3]
        return T
    return None


def load_extrinsics_csv(path):
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        print(f"[calib] WARNING: EXTRINSICS_CSV not found: {p}. Proceeding without extrinsics.")
        return {}

    with p.open(newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].lstrip().startswith("#")]
    if not rows:
        print(f"[calib] WARNING: EXTRINSICS_CSV is empty: {p}")
        return {}

    def numeric(row):
        try:
            [float(c) for c in row if c.strip() != ""]
            return True
        except ValueError:
            return False

    header, start = None, 0
    if not numeric(rows[0]):
        header = [c.strip().lower() for c in rows[0]]
        start = 1

    frame_col = 0
    pose_cols = None
    if header:
        for i, name in enumerate(header):
            if name in ("frame", "frame_id", "frameid", "index", "idx", "id", "n"):
                frame_col = i
                break
        named = {n: i for i, n in enumerate(header)}
        rot9 = [f"r{i}{j}" for i in range(3) for j in range(3)]
        trans = next((t for t in (("tx", "ty", "tz"), ("x", "y", "z"))
                      if set(t) <= named.keys()), None)
        if set(rot9) <= named.keys() and trans:
            pose_cols = [named[k] for k in
                         ("r00", "r01", "r02", trans[0],
                          "r10", "r11", "r12", trans[1],
                          "r20", "r21", "r22", trans[2])]
        elif {"tx", "ty", "tz", "qx", "qy", "qz", "qw"} <= named.keys():
            pose_cols = [named[k] for k in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")]
        elif {"x", "y", "z", "qx", "qy", "qz", "qw"} <= named.keys():
            pose_cols = [named[k] for k in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        elif {"x", "y", "z", "roll", "pitch", "yaw"} <= named.keys():
            pose_cols = [named[k] for k in ("x", "y", "z", "roll", "pitch", "yaw")]

    cam_from_body = np.asarray(CAM_FROM_BODY, np.float64) if CAM_FROM_BODY is not None else None

    poses = {}
    for row in rows[start:]:
        cells = [c for c in row if c.strip() != ""]
        try:
            frame = int(float(cells[frame_col]))
            vals = ([float(cells[i]) for i in pose_cols] if pose_cols else
                    [float(c) for j, c in enumerate(cells) if j != frame_col])
        except (ValueError, IndexError):
            continue

        T = _pose_row_to_matrix(vals)
        if T is None or not np.all(np.isfinite(T)):
            continue
        if cam_from_body is not None:
            A = np.eye(4)
            A[:3, :3] = cam_from_body
            T = T @ A.T if EXTRINSICS_CONVENTION == "c2w" else A @ T
        if EXTRINSICS_CONVENTION == "c2w":
            T = _invert_rt(T)
        poses[frame] = T.astype(np.float32)

    return poses


def poses_alignable(exts, tol=1e-3):
    if len(exts) < 3:
        return False
    if not all(np.all(np.isfinite(T)) for T in exts):
        return False
    centres = np.stack([-T[:3, :3].T @ T[:3, 3] for T in exts])
    sv = np.linalg.svd(centres - centres.mean(0), compute_uv=False)
    return len(sv) >= 2 and sv[0] > 0 and sv[1] > tol * sv[0]


def pose_for_frame(poses, idx):
    if not poses:
        return None
    if idx in poses:
        return poses[idx]
    near = min(poses, key=lambda k: abs(k - idx))
    return poses[near] if abs(near - idx) <= POSE_MAX_FRAME_GAP else None


# ------------------------- MODEL INFERENCE -------------------------

def _bootstrap_da3():
    try:
        from depth_anything_3.api import DepthAnything3
        return DepthAnything3
    except ModuleNotFoundError:
        here = Path(__file__).resolve().parent
        last_err = None
        for local in (here / "depth-anything-3" / "src",
                      here / "depth-anything-3" / "depth-anything-3" / "src"):
            if local.is_dir():
                sys.path.insert(0, str(local))
                try:
                    from depth_anything_3.api import DepthAnything3
                    return DepthAnything3
                except ModuleNotFoundError as exc:
                    last_err = exc
                    sys.path.remove(str(local))
        raise SystemExit(f"depth_anything_3 not importable: {last_err}")


def load_model():
    DepthAnything3 = _bootstrap_da3()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "cuda" in device:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[load] {MODEL_NAME} -> {device}")
    model = DepthAnything3.from_pretrained(MODEL_NAME).to(device).eval()
    return model, device


_WARNED = set()

def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        print(msg)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return None if x is None else np.asarray(x)


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def sanitize_depth(depth):
    d = np.asarray(depth, np.float32)
    good = np.isfinite(d) & (d > 0)
    if not good.all():
        fill = float(np.median(d[good])) if good.any() else 1.0
        d = np.where(good, d, fill).astype(np.float32)
    return d, good


@torch.inference_mode()
def predict(model, frame_bgr, K=None, ext=None, support=None):
    views = list(support or [])
    rgbs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f, _ in views]
    rgbs.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    exts = [T for _, T in views] + [ext]
    kwargs = {}
    if all(T is not None for T in exts) and poses_alignable(exts):
        kwargs["extrinsics"] = np.stack(exts).astype(np.float32)
    if K is not None:
        kwargs["intrinsics"] = np.repeat(K[None], len(rgbs), axis=0).astype(np.float32)

    def run(imgs, kw, res):
        return model.inference(imgs, export_dir=None, process_res=res, **kw)

    res = PROCESS_RES
    try:
        pred = run(rgbs, kwargs, res)
    except Exception as exc:
        if _is_oom(exc) and OOM_RETRY and len(rgbs) > 1:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rgbs, kwargs = rgbs[-1:], {k: v[-1:] for k, v in kwargs.items()}
            kwargs.pop("extrinsics", None)
            pred = run(rgbs, kwargs, res)
        elif "extrinsics" in kwargs and "umeyama" in str(exc).lower():
            kwargs.pop("extrinsics")
            pred = run(rgbs, kwargs, res)
        else:
            raise

    target = len(rgbs) - 1
    h, w = frame_bgr.shape[:2]
    
    raw_depth = _to_numpy(getattr(pred, "depth", None))
    if raw_depth is None:
        raise RuntimeError("Model returned no depth output.")
    
    depth = np.squeeze(raw_depth[target]).astype(np.float32)
    dw = depth.shape[1]
    
    # Clean upsampling without incorrect radial transformation
    depth, good = sanitize_depth(cv2.resize(depth, (w, h), interpolation=UPSAMPLE))

    def _mask(arr):
        arr = _to_numpy(arr)
        if arr is None:
            return None
        arr_target = np.squeeze(arr[target]).astype(np.float32)
        return cv2.resize(arr_target, (w, h), interpolation=cv2.INTER_NEAREST)

    intr = _to_numpy(getattr(pred, "intrinsics", None))
    focal_px = None
    if intr is not None and intr.size >= 9:
        focal_px = float(np.asarray(intr).reshape(-1, 3, 3)[target][0, 0]) * (w / float(dw))

    return {
        "depth": depth,
        "finite": good,
        "conf": _mask(getattr(pred, "conf", None)),
        "sky": _mask(getattr(pred, "sky", None)),
        "focal_px": focal_px,
        "is_metric": bool(getattr(pred, "is_metric", 0)),
        "pose_scaled": "extrinsics" in kwargs,
        "n_views": len(rgbs),
        "process_res": res,
        "used_K": "intrinsics" in kwargs,
        "used_ext": "extrinsics" in kwargs,
    }


def valid_mask(out, use_conf, use_sky):
    m = out["finite"].copy()
    if use_sky and out["sky"] is not None:
        m &= out["sky"] < 0.5
    if use_conf and out["conf"] is not None:
        c = out["conf"]
        finite = np.isfinite(c)
        if finite.any():
            m &= c >= float(np.percentile(c[finite], CONF_PERCENTILE))
    return m if m.mean() > 0.02 else out["finite"].copy()


# --------------------------- VISUALIZATION ---------------------------

def color_field(depth):
    d = np.clip(depth, 1e-6, None)
    return (1.0 / d).astype(np.float32) if COLOR_ON_DISPARITY else d.astype(np.float32)


def _fit_sample(field, mask, cap=400_000):
    vals = field[mask] if (mask is not None and mask.any()) else field.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros(1, np.float32)
    if vals.size > cap:
        vals = vals[:: int(np.ceil(vals.size / cap))]
    return vals


def normalize_frame(field, mask, mode, gamma, smooth, pct=DISP_PCT, depth=None, K=None):
    sample = _fit_sample(field, mask)

    if mode == "rank":
        vmin, vmax = float(sample.min()), float(sample.max())
        if vmax - vmin < 1e-12:
            norm = np.zeros_like(field)
        else:
            bins = 1 << 16
            scale = (bins - 1) / (vmax - vmin)
            hist = np.bincount(
                np.clip((sample - vmin) * scale, 0, bins - 1).astype(np.int64),
                minlength=bins)
            cdf = np.cumsum(hist).astype(np.float32)
            cdf /= max(cdf[-1], 1.0)
            idx_clip = np.clip((field - vmin) * scale, 0, bins - 1).astype(np.int64)
            norm = cdf[idx_clip]
    else:
        lo, hi = (float(v) for v in np.percentile(sample, pct))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            lo, hi = float(sample.min()), float(sample.min()) + 1e-6
        norm = np.clip((field - lo) / (hi - lo), 0.0, 1.0)
        vmin, vmax = lo, hi

    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
    if smooth:
        norm = cv2.bilateralFilter(norm, BILATERAL_D, BILATERAL_COLOR, BILATERAL_SPACE)
    return norm, vmin, vmax


_LUT_CACHE = {}

def _lut(cmap, levels):
    key = (cmap, levels)
    if key not in _LUT_CACHE:
        base = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1),
                                 cmap).reshape(256, 3).astype(np.float32)
        xs = np.linspace(0, 255, levels)
        _LUT_CACHE[key] = np.stack(
            [np.interp(xs, np.arange(256), base[:, c]) for c in range(3)],
            axis=1).astype(np.uint8)
    return _LUT_CACHE[key]


def colorize(norm, mask=None, cmap=COLORMAP, levels=CMAP_LEVELS):
    lut = _lut(cmap, levels)
    idx = np.clip(norm * (levels - 1), 0, levels - 1).astype(np.int32)
    img = lut[idx]
    if DIM_INVALID and mask is not None:
        img = img.copy()
        img[~mask] = (30, 30, 30)
    return np.ascontiguousarray(img)


def main():
    model, device = load_model()

    src = 0 if VIDEO_PATH in ("0", 0) else VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {VIDEO_PATH}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    ok, frame = cap.read()
    if not ok or frame is None:
        raise SystemExit("Empty or unreadable video file.")
    fh, fw = frame.shape[:2]

    K = load_intrinsics(INTRINSICS, (fw, fh), INTRINSICS_RES)
    poses = load_extrinsics_csv(EXTRINSICS_CSV) if EXTRINSICS_CSV else {}

    start = 0
    if src != 0 and START_FRAME > 0:
        start = min(int(START_FRAME), max(total - 1, 0)) if total else int(START_FRAME)
        print(f"[video] starting at frame {start}" + (f" / {total}" if total else ""))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    win = "Depth Anything V3 - Fixed"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps_smooth, idx = 0.0, start
    history = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            idx += 1
            if FRAME_STRIDE > 1 and idx % FRAME_STRIDE:
                continue

            pose = pose_for_frame(poses, idx + POSE_FRAME_OFFSET)
            history.append((idx, frame, pose))
            span = max(WINDOW_SIZE - 1, 0) * max(WINDOW_STRIDE, 1) + 1
            del history[:-span]

            support = []
            if WINDOW_SIZE > 1 and pose is not None:
                for back in range(WINDOW_SIZE - 1, 0, -1):
                    want = idx - back * WINDOW_STRIDE
                    hit = next((h for h in history if h[0] == want), None)
                    if hit and hit[2] is not None:
                        support.append((hit[1], hit[2]))

            t0 = time.perf_counter()
            out = predict(model, frame, K, pose, support)
            if "cuda" in device:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            fps_smooth = (1.0 / max(dt, 1e-6)) if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * (1.0 / max(dt, 1e-6))

            mask = valid_mask(out, USE_CONF_MASK, USE_SKY_MASK)
            norm, vmin, vmax = normalize_frame(color_field(out["depth"]), mask,
                                               NORM_MODE, DISPLAY_GAMMA, SPATIAL_SMOOTH,
                                               DISP_PCT, depth=out["depth"], K=K)

            depth_color = colorize(norm, mask, COLORMAP)
            display = np.hstack([frame, depth_color]) if VIEW_MODE == "side" else depth_color

            cv2.imshow(win, display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
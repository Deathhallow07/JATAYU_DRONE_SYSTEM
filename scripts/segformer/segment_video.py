"""
Live navigation-ontology segmentation on video.

Runs the multi-dataset SegFormer (MiT-B3) over a video file or camera stream and shows,
in real time: the raw frame, the colorized semantic mask, the blended overlay, a binary
traversable/non-traversable view, and a legend with live per-class pixel coverage.

The class ontology, colors and traversability grouping are imported from
SegFormer_training.py so inference can never drift from what the model was trained on.

Usage
-----
  python segment_video.py --video input.mp4
  python segment_video.py --video input.mp4 --view overlay --save-dir out/ 
  python segment_video.py --video 0 --stride 2            # webcam, infer every 2nd frame
  python segment_video.py --video in.mp4 --no-display --save-dir out/   # headless render

Keys (while the window is focused)
  q / ESC  quit             space  pause            s  save current frame as PNG
  g  grid view              1  original   2  mask   3  overlay   4  traversability
  [ / ]  overlay alpha down/up                      l  toggle legend
"""

import os
import sys
import time
import argparse

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Single source of truth for the ontology — importing rather than re-declaring means a
# class or color edit in the training script can never silently desync from inference.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from SegFormer_training import (
        CLASS_NAMES, CLASS_COLORS, NUM_CLASSES, TRAVERSABLE_IDS, UNUSED_CLASSES,
    )
except ImportError as e:
    raise SystemExit(
        "Could not import the ontology from SegFormer_training.py — keep segment_video.py "
        f"in the same directory as the training script. ({e})"
    )

DEFAULT_CKPT = "./segformer_b3_multidataset/stage3_uavid"

# cv2 works in BGR; CLASS_COLORS is authored RGB.
CLASS_COLORS_BGR = CLASS_COLORS[:, ::-1].copy()
TRAV_MASK = np.zeros(NUM_CLASSES, dtype=bool)
TRAV_MASK[TRAVERSABLE_IDS] = True
TRAV_COLORS_BGR = np.array([(60, 200, 60) if TRAV_MASK[i] else (60, 60, 220)
                            for i in range(NUM_CLASSES)], dtype=np.uint8)

FONT = cv2.FONT_HERSHEY_SIMPLEX
LEGEND_WIDTH = 300


# ==========================================
# Model
# ==========================================
class Segmenter:
    """Wraps the trained SegFormer. Frames are letterbox-free: they are resized to a square
    inference resolution exactly as validation did, then the logits are bilinearly upsampled
    back to the output resolution before argmax — upsampling logits rather than the argmax
    keeps class boundaries smooth instead of blocky."""

    def __init__(self, ckpt, device, infer_size=768, use_half=True):
        self.device = torch.device(device)
        self.infer_size = infer_size
        self.model = SegformerForSemanticSegmentation.from_pretrained(ckpt).to(self.device).eval()

        proc = SegformerImageProcessor.from_pretrained(ckpt)
        self.mean = torch.tensor(proc.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(proc.image_std, device=self.device).view(1, 3, 1, 1)

        n = self.model.config.num_labels
        if n != NUM_CLASSES:
            raise SystemExit(f"Checkpoint has {n} classes but the ontology defines {NUM_CLASSES}.")

        self.dtype = torch.float32
        if use_half and self.device.type == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    @torch.inference_mode()
    def __call__(self, frame_bgr, out_hw):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.infer_size, self.infer_size), interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        x = (x - self.mean) / self.std

        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            logits = self.model(pixel_values=x.to(self.dtype)).logits

        logits = F.interpolate(logits.float(), size=out_hw, mode="bilinear", align_corners=False)
        # Class 0 is the Ignore slot and was never a training target; suppressing it stops
        # stray "unknown" speckle from ever reaching the cost map.
        for c in UNUSED_CLASSES:
            logits[:, c] = -1e4
        return logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()


# ==========================================
# Rendering
# ==========================================
def colorize(label, palette):
    return palette[label]


def blend(frame, color_mask, alpha):
    return cv2.addWeighted(frame, 1.0 - alpha, color_mask, alpha, 0.0)


def class_coverage(label):
    counts = np.bincount(label.ravel(), minlength=NUM_CLASSES).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def draw_legend(height, coverage, alpha, fps, trav_pct):
    """Legend panel: swatch, class name, and live pixel coverage. Classes absent from the
    current frame are dimmed rather than hidden, so rows never jump around between frames."""
    panel = np.full((height, LEGEND_WIDTH, 3), 28, dtype=np.uint8)
    cv2.putText(panel, "NAVIGATION CLASSES", (14, 30), FONT, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.line(panel, (14, 42), (LEGEND_WIDTH - 14, 42), (80, 80, 80), 1)

    y = 68
    for i in range(NUM_CLASSES):
        if i in UNUSED_CLASSES:
            continue
        pct = coverage[i] * 100.0
        present = pct >= 0.05
        color = tuple(int(c) for c in CLASS_COLORS_BGR[i])
        if not present:
            color = tuple(int(c * 0.35) for c in color)
        cv2.rectangle(panel, (14, y - 12), (36, y + 4), color, -1)
        cv2.rectangle(panel, (14, y - 12), (36, y + 4), (90, 90, 90), 1)

        text_col = (235, 235, 235) if present else (110, 110, 110)
        marker = "+" if TRAV_MASK[i] else "x"  # + drivable, x obstacle
        cv2.putText(panel, f"{marker} {CLASS_NAMES[i]}", (46, y), FONT, 0.44, text_col, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{pct:5.1f}%", (LEGEND_WIDTH - 68, y), FONT, 0.42, text_col, 1, cv2.LINE_AA)
        y += 26

    y += 6
    cv2.line(panel, (14, y), (LEGEND_WIDTH - 14, y), (80, 80, 80), 1)
    y += 26
    cv2.putText(panel, "+ drivable   x obstacle", (14, y), FONT, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    y += 26
    cv2.putText(panel, f"Traversable: {trav_pct * 100:.1f}%", (14, y), FONT, 0.46,
                (80, 220, 80), 1, cv2.LINE_AA)
    y += 26
    cv2.putText(panel, f"{fps:.1f} FPS   alpha {alpha:.2f}", (14, y), FONT, 0.44,
                (170, 170, 170), 1, cv2.LINE_AA)
    return panel


def label_tile(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 11 + 16, 30), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 21), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def compose(view, frame, mask_c, overlay, trav_c):
    """Build the image body for the requested view mode."""
    if view == "grid":
        h, w = frame.shape[:2]
        half = (w // 2, h // 2)
        tiles = [
            label_tile(cv2.resize(frame, half), "INPUT"),
            label_tile(cv2.resize(mask_c, half, interpolation=cv2.INTER_NEAREST), "SEMANTIC MASK"),
            label_tile(cv2.resize(overlay, half), "OVERLAY"),
            label_tile(cv2.resize(trav_c, half, interpolation=cv2.INTER_NEAREST), "TRAVERSABILITY"),
        ]
        return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    return label_tile({"original": frame, "mask": mask_c,
                       "overlay": overlay, "traversability": trav_c}[view], view.upper())


# ==========================================
# Main loop
# ==========================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Path to a video file, or a camera index like 0.")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help=f"Checkpoint directory (default: {DEFAULT_CKPT})")
    ap.add_argument("--view", default="grid",
                    choices=["grid", "original", "mask", "overlay", "traversability"])
    ap.add_argument("--infer-size", type=int, default=768, help="Network input resolution (matches validation).")
    ap.add_argument("--max-width", type=int, default=1280, help="Output/display width; frames are scaled to fit.")
    ap.add_argument("--alpha", type=float, default=0.55, help="Overlay blend strength.")
    ap.add_argument("--stride", type=int, default=1, help="Run the network every Nth frame, reusing the last mask.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true", help="Disable half/bfloat16 inference.")
    ap.add_argument("--save-dir", default=None, help="Write mask/overlay/traversability/grid videos here.")
    ap.add_argument("--no-display", action="store_true", help="Render without opening a window.")
    ap.add_argument("--no-legend", action="store_true")
    args = ap.parse_args()

    source = int(args.video) if args.video.isdigit() else args.video
    if isinstance(source, str) and not os.path.exists(source):
        raise SystemExit(f"Video not found: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.video}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = min(1.0, args.max_width / max(src_w, 1))
    out_w, out_h = int(src_w * scale) // 2 * 2, int(src_h * scale) // 2 * 2

    print(f"Loading {args.ckpt} on {args.device} ...")
    seg = Segmenter(args.ckpt, args.device, args.infer_size, use_half=not args.fp32)
    print(f"Source {src_w}x{src_h} @ {src_fps:.1f} FPS"
          f"{f' ({total} frames)' if total > 0 else ''} -> processing at {out_w}x{out_h}")

    writers = {}
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        grid_size = (out_w, out_h)  # grid tiles are half-size, so the mosaic matches the frame
        for tag, size in (("mask", (out_w, out_h)), ("overlay", (out_w, out_h)),
                          ("traversability", (out_w, out_h)), ("grid", grid_size)):
            writers[tag] = cv2.VideoWriter(os.path.join(args.save_dir, f"{tag}.mp4"),
                                           fourcc, src_fps, size)
        print(f"Writing videos to {args.save_dir}/")

    view, alpha, paused, show_legend = args.view, args.alpha, False, not args.no_legend
    label = None
    fps_ema, frame_idx, saved = 0.0, 0, 0
    window = "Navigation Segmentation"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                t0 = time.time()
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                # --stride reuses the previous mask on skipped frames; the video still plays
                # at full rate while the network runs at a fraction of it.
                if label is None or frame_idx % args.stride == 0:
                    label = seg(frame, (out_h, out_w))

                mask_c = colorize(label, CLASS_COLORS_BGR)
                trav_c = colorize(label, TRAV_COLORS_BGR)
                overlay = blend(frame, mask_c, alpha)

                cov = class_coverage(label)
                trav_pct = float(cov[TRAVERSABLE_IDS].sum())

                dt = time.time() - t0
                fps_ema = (1.0 / dt) if fps_ema == 0 else 0.9 * fps_ema + 0.1 * (1.0 / max(dt, 1e-6))
                frame_idx += 1

            body = compose(view, frame, mask_c, overlay, trav_c)
            if show_legend:
                canvas = np.hstack([body, draw_legend(body.shape[0], cov, alpha, fps_ema, trav_pct)])
            else:
                canvas = body

            for tag, w in writers.items():
                w.write({"mask": mask_c, "overlay": overlay,
                         "traversability": trav_c, "grid": compose("grid", frame, mask_c, overlay, trav_c)}[tag])

            if total > 0 and frame_idx % 30 == 0:
                print(f"\r  frame {frame_idx}/{total} ({frame_idx / total * 100:5.1f}%)  "
                      f"{fps_ema:.1f} FPS", end="", flush=True)

            if not args.no_display:
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("g"):
                    view = "grid"
                elif key == ord("1"):
                    view = "original"
                elif key == ord("2"):
                    view = "mask"
                elif key == ord("3"):
                    view = "overlay"
                elif key == ord("4"):
                    view = "traversability"
                elif key == ord("l"):
                    show_legend = not show_legend
                elif key == ord("["):
                    alpha = max(0.0, alpha - 0.05)
                elif key == ord("]"):
                    alpha = min(1.0, alpha + 0.05)
                elif key == ord("s"):
                    p = os.path.join(args.save_dir or ".", f"frame_{frame_idx:06d}.png")
                    cv2.imwrite(p, canvas)
                    saved += 1
                    print(f"\n  saved {p}")
    finally:
        cap.release()
        for w in writers.values():
            w.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"\nDone. {frame_idx} frames at {fps_ema:.1f} FPS."
          + (f" Outputs in {args.save_dir}/" if args.save_dir else "")
          + (f" {saved} stills saved." if saved else ""))


if __name__ == "__main__":
    main()

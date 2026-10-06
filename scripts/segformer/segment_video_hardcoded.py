"""
Live navigation-ontology segmentation on video.

Runs the multi-dataset SegFormer (MiT-B3) over a video file or camera stream and shows,
in real time: the raw frame, the colorized semantic mask, the blended overlay, a binary
traversable/non-traversable view, and a legend with live per-class pixel coverage.

The class ontology, colors and traversability grouping are imported from
SegFormer_training.py so inference can never drift from what the model was trained on.

Configuration
-------------
Edit the paths and options in the CONFIG section below before running the script.
- INPUT_PT_PATH: path to the trained SegFormer checkpoint directory
- INPUT_VIDEO_PATH: path to the input video
- SAVE_VIDEO_PATH: optional output video path; set to None to disable saving


Keys (while the window is focused)
  q / ESC  quit             space  pause            s  save current frame as PNG
  g  grid view              1  original   2  mask   3  overlay   4  traversability
  [ / ]  overlay alpha down/up                      l  toggle legend
"""

import os
import sys
import time

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

# ==========================================
# CONFIG
# ==========================================
# Edit these paths/options instead of passing command-line arguments.

INPUT_PT_PATH = "/media/uasdtu/DataSets2/Segmentation_GATE_1/segformer_b3_multidataset/stage3_uavid" 
INPUT_VIDEO_PATH = "/media/uasdtu/DataSets3/dtu_tour.mp4"
SAVE_VIDEO_PATH = None  # e.g. "./segmented_output.mp4", or None to disable saving

VIEW = "grid"  # "grid", "original", "mask", "overlay", "traversability"
INFER_SIZE = 768
MAX_WIDTH = 1280
ALPHA = 0.55
STRIDE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = True
SHOW_DISPLAY = True
SHOW_LEGEND = True

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
    ckpt = INPUT_PT_PATH
    video_path = INPUT_VIDEO_PATH
    save_video_path = SAVE_VIDEO_PATH

    view = VIEW
    infer_size = INFER_SIZE
    max_width = MAX_WIDTH
    alpha = ALPHA
    stride = STRIDE
    device = DEVICE
    use_half = USE_HALF
    show_display = SHOW_DISPLAY
    show_legend = SHOW_LEGEND

    source = int(video_path) if str(video_path).isdigit() else video_path
    source = int(video_path) if video_path.isdigit() else video_path
    if isinstance(source, str) and not os.path.exists(source):
        raise SystemExit(f"Video not found: {source}")

    save_dir = None
    if save_video_path:
        save_dir = os.path.dirname(os.path.abspath(save_video_path)) or "."
        os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = min(1.0, max_width / max(src_w, 1))
    out_w, out_h = int(src_w * scale) // 2 * 2, int(src_h * scale) // 2 * 2

    print(f"Loading {ckpt} on {device} ...")
    seg = Segmenter(ckpt, device, infer_size, use_half=not (not use_half))
    print(f"Source {src_w}x{src_h} @ {src_fps:.1f} FPS"
          f"{f' ({total} frames)' if total > 0 else ''} -> processing at {out_w}x{out_h}")

    writer = None
    if save_video_path:
        print(f"Output video will be written to {save_video_path}")


    paused = False
    label = None
    fps_ema, frame_idx, saved = 0.0, 0, 0
    window = "Navigation Segmentation"
    if not (not show_display):
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                t0 = time.time()
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                # STRIDE reuses the previous mask on skipped frames; the video still plays
                # at full rate while the network runs at a fraction of it.
                if label is None or frame_idx % stride == 0:
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

            if writer is None and save_video_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    save_video_path, fourcc, src_fps,
                    (canvas.shape[1], canvas.shape[0])
                )
                if not writer.isOpened():
                    raise SystemExit(f"Could not open output video for writing: {save_video_path}")

            if writer is not None:
                # Save exactly what is displayed, including the legend when enabled.
                # If the legend is enabled, the output width is adjusted below.
                writer.write(canvas)

            if total > 0 and frame_idx % 30 == 0:
                print(f"\r  frame {frame_idx}/{total} ({frame_idx / total * 100:5.1f}%)  "
                      f"{fps_ema:.1f} FPS", end="", flush=True)

            if not (not show_display):
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
                    p = os.path.join(save_dir or ".", f"frame_{frame_idx:06d}.png")
                    cv2.imwrite(p, canvas)
                    saved += 1
                    print(f"\n  saved {p}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not (not show_display):
            cv2.destroyAllWindows()

    print(f"\nDone. {frame_idx} frames at {fps_ema:.1f} FPS."
          + (f" Output video: {save_video_path}" if save_video_path else "")
          + (f" {saved} stills saved." if saved else ""))


if __name__ == "__main__":
    main()

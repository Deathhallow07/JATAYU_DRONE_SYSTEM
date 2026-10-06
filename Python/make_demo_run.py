#!/usr/bin/env python3
"""
Generate a synthetic run folder — no video, no models, no GPU.

The console reads artifacts, so a folder in the pipeline's format is all it
needs to come up fully populated. Use this to develop the UI, to rehearse, or
to demo on a machine with nothing installed.

Everything here is fabricated, including the crops. For a demo on real footage
use ingest_video.py instead, which runs YOLOE over an actual video.

    python make_demo_run.py --out ../demo_runs --gids 6
    python mission_watcher.py --root ../demo_runs --replay --speed 4
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_writer import RunWriter  # noqa: E402

# Roughly DTU, Delhi — somewhere plausible for the coordinates to sit.
ORIGIN = (28.7501, 77.1177)


def synth_crop(rng, w=128, h=256, seed_hue=0):
    """A crude person-shaped crop: warm figure on a cool ground, plus noise."""
    img = np.full((h, w, 3), 90 + seed_hue % 40, np.uint8)
    img = cv2.add(img, rng.integers(0, 45, (h, w, 3), dtype=np.uint8))
    cv2.ellipse(img, (w // 2, int(h * 0.22)), (w // 6, h // 12), 0, 0, 360, (150, 170, 200), -1)
    cv2.rectangle(img, (w // 4, int(h * 0.3)), (3 * w // 4, int(h * 0.78)), (120, 140, 190), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "demo_runs"))
    ap.add_argument("--gids", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    w = RunWriter(args.out)

    w.log("INFO", f"Mission Folder: {w.run}")
    w.log("OK", "YOLOE detector loaded (yoloe-26x-seg.pt)")
    w.log("OK", "ReID model loaded (epoch_003.pth)")
    w.log("INFO", "Telemetry source: csv")
    w.log("INFO", "Video source: video")
    w.log("WARN", "SYNTHETIC RUN — no video, no models, fabricated artifacts")

    frame = 0
    total_entries = 0

    for gid in range(1, args.gids + 1):
        # Interleave chatter with the identity events so the replayed log reads
        # like a mission rather than a burst of GIDs at the end.
        for _ in range(int(rng.integers(40, 110))):
            frame += 1
            w.log("FRAME", f"FPS={10 + rng.random() * 8:.1f}")
            if rng.random() < 0.28:
                w.log("DET", f"frame={frame} raw={rng.integers(0, 4)} final={rng.integers(0, 3)}")
            if rng.random() < 0.12:
                w.log("TRACK", f"tid={rng.integers(1, 30)} updated len={rng.integers(5, 60)}")
            if rng.random() < 0.05:
                w.log("WARN", f"frame={frame} blur reject (sharpness={rng.random() * 60:.1f})")

        lat = round(ORIGIN[0] + float(rng.normal(0, 4e-4)), 8)
        lon = round(ORIGIN[1] + float(rng.normal(0, 4e-4)), 8)

        # A second buffer means the pipeline re-identified this person later —
        # the interesting case, so make it common but not universal.
        n_buffers = 1 if rng.random() < 0.55 else 2
        cumulative = 0
        first_frame = frame

        for b in range(n_buffers):
            size = int(rng.integers(8, 18))
            base = frame + b * int(rng.integers(150, 500))

            entries, crops = [], {}
            for j in range(size):
                f = base + j
                entries.append((
                    f,
                    float(np.clip(rng.normal(0.82, 0.07), 0.3, 0.99)),
                    float(rng.uniform(80, 420)),
                    round(lat + float(rng.normal(0, 8e-6)), 8),
                    round(lon + float(rng.normal(0, 8e-6)), 8),
                ))
                crops[f] = synth_crop(rng, seed_hue=gid * 17)

            cumulative += size
            total_entries += size

            w.write_buffer(
                gid=gid,
                event="new" if b == 0 else "merge",
                tid=int(rng.integers(1, 40)),
                entries=entries,
                crops=crops,
                lat=lat,
                lon=lon,
                cumulative=cumulative,
                first_frame=first_frame,
                ros_time=base / 15.0,
                representative=synth_crop(rng, w=160, h=320, seed_hue=gid * 17),
            )

        frame += 60

    w.log("INFO", "=" * 40)
    w.log("INFO", "MISSION COMPLETE")
    w.log("INFO", f"Frames               : {frame}")
    w.log("INFO", f"GIDs created         : {args.gids}")
    w.log("INFO", f"Gallery entries      : {total_entries}")
    w.flush_log()

    print(f"run folder : {w.run}")
    print(f"gids       : {args.gids}")
    print()
    print("Replay it with:")
    print(f"  python mission_watcher.py --root {os.path.abspath(args.out)} --replay --speed 4")


if __name__ == "__main__":
    main()

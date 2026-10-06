#!/usr/bin/env python3
"""
Depth pane worker.

Two backends:

    --backend stub   (default) a gradient/texture heuristic. Not depth — it is
                     a placeholder with the right shape and the right output
                     contract so the pane, the relay and the layout can be
                     built and demoed before the weights land.

    --backend dav2   Depth-Anything-V2 via transformers. Everything real lives
                     in _load_dav2/_infer_dav2 below; nothing else in the app
                     changes when you switch.

The published payload is always the same:

    jpeg        colour-mapped relative depth
    depthMin/Max, depthMean   scalar summary for the pane's readout
    backend     which of the above produced it

so the pane cannot tell the difference and does not need to.
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import BaseWorker, label_frame, normalize01  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("depth")

# Checkpoint sizes, smallest first. Small is the one to demo on a laptop.
DAV2_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthWorker(BaseWorker):
    NAME = "depth"

    def __init__(self, *args, backend="stub", model="small", colormap="inferno", **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.model_size = model
        self.colormap = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_INFERNO)
        self._pipe = None
        self._device = "cpu"

    # -- load --------------------------------------------------------------

    def load(self):
        if self.backend == "stub":
            log.warning("running the STUB depth backend — output is not real depth")
            return
        self._load_dav2()

    def _load_dav2(self):
        """
        Real backend. Requires:  pip install transformers torch pillow
        """
        import torch
        from transformers import pipeline

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        repo = DAV2_MODELS.get(self.model_size, self.model_size)

        log.info("loading %s on %s", repo, self._device)
        self._pipe = pipeline(
            task="depth-estimation",
            model=repo,
            device=0 if self._device == "cuda" else -1,
        )
        log.info("depth model ready")

    # -- infer -------------------------------------------------------------

    def infer(self, frame):
        depth = (self._infer_stub(frame) if self.backend == "stub"
                 else self._infer_dav2(frame))

        norm = normalize01(depth)
        vis = cv2.applyColorMap((norm * 255).astype(np.uint8), self.colormap)

        # Blend a little of the source back in. A bare colormap is pretty but
        # unreadable on stage — the operator cannot tell which blob is the
        # casualty. At 0.25 the scene stays recognisable.
        vis = cv2.addWeighted(vis, 0.75, frame, 0.25, 0)

        tag = "DEPTH · STUB" if self.backend == "stub" else f"DEPTH · DAv2-{self.model_size} · {self._device}"
        label_frame(vis, tag, (120, 220, 255))

        return vis, {
            "backend": self.backend,
            "device": self._device,
            "depthMin": float(depth.min()),
            "depthMax": float(depth.max()),
            "depthMean": float(depth.mean()),
        }

    def _infer_dav2(self, frame):
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = self._pipe(Image.fromarray(rgb))
        depth = np.asarray(out["predicted_depth"] if "predicted_depth" in out else out["depth"],
                           dtype=np.float32)

        # The pipeline may return the model's native resolution rather than the
        # input's; the pane overlays this on the source frame, so it must match.
        if depth.shape[:2] != frame.shape[:2]:
            depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        return depth

    @staticmethod
    def _infer_stub(frame):
        """
        Placeholder. Combines two cues that correlate loosely with depth in an
        aerial frame — vertical position (further up the frame is further away)
        and local texture energy (near surfaces resolve more detail) — then
        smooths. Good enough to show the pane working; not a depth estimate.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        vertical = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None].repeat(w, axis=1)

        texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        texture = cv2.GaussianBlur(np.abs(texture), (0, 0), 9)
        texture = normalize01(texture)

        depth = 0.65 * vertical + 0.35 * texture
        return cv2.GaussianBlur(depth, (0, 0), 5)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, image, camera index, or MJPEG URL")
    ap.add_argument("--backend", choices=["stub", "dav2"], default="stub")
    ap.add_argument("--model", default="small", choices=list(DAV2_MODELS) , help="dav2 checkpoint size")
    ap.add_argument("--colormap", default="inferno")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--url", default=os.environ.get("GUI_URL", "http://127.0.0.1:7100"))
    args = ap.parse_args()

    DepthWorker(
        args.source, url=args.url, fps=args.fps, loop=not args.no_loop, width=args.width,
        backend=args.backend, model=args.model, colormap=args.colormap,
    ).start()


if __name__ == "__main__":
    main()

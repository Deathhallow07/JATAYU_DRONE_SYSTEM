#!/usr/bin/env python3
"""
Traversability pane worker — segmentation -> cost grid -> responder route.

Three stages, and only the first is a placeholder:

  1. SEGMENT   frame -> traversable / non-traversable mask.
               --backend stub      colour + texture heuristic (placeholder)
               --backend segformer your SegFormer checkpoint (real; see
                                   _load_segformer)

  2. COST      mask -> coarse grid. Each cell is free or blocked, and free
               cells are weighted by their distance to the nearest obstacle,
               so the planner prefers the middle of an open route over
               scraping a wall. This is real and backend-independent.

  3. PLAN      A* over that grid from the responder's start to the casualty.
               Real. 8-connected, octile heuristic, clearance-weighted.

The point of the pane is stage 3 — showing that once the scene is segmented,
the route to a located casualty falls out. Stages 1 and 2 can be swapped
without touching it.

Geometry note: the grid is in IMAGE space, not world space. Turning a GID's
lat/lon into a grid cell needs the pipeline's CoordinateTransformer and the
frame's telemetry; until that is wired, the goal comes from the UI (click the
map) or from --goal. See on_command().
"""

import argparse
import heapq
import logging
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import BaseWorker, label_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seg")

BLOCKED = 1
FREE = 0

# 8-connected neighbourhood; diagonals cost sqrt(2) so the planner cannot buy
# distance by zig-zagging.
NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
              (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def astar(grid, cost, start, goal):
    """
    A* on a boolean occupancy grid with a per-cell traversal multiplier.

    grid  (H,W) uint8, BLOCKED/FREE
    cost  (H,W) float32, >= 1.0, multiplier applied on entering a cell
    start,goal  (row, col)

    Returns a list of (row, col) including both ends, or [] if unreachable.

    Octile heuristic — admissible for 8-connected movement with these step
    costs, so the first time goal is popped the path is optimal. (Euclidean
    would also be admissible but expands more nodes; Manhattan would NOT be,
    because diagonals are cheaper than two straight steps.)
    """
    h, w = grid.shape

    def ok(rc):
        r, c = rc
        return 0 <= r < h and 0 <= c < w and grid[r, c] == FREE

    if not ok(start) or not ok(goal):
        return []

    def heuristic(a, b):
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)

    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        r, c = current
        for dr, dc, step in NEIGHBOURS:
            nb = (r + dr, c + dc)
            if nb in closed or not ok(nb):
                continue
            # No corner-cutting: a diagonal is only legal when both orthogonal
            # neighbours it passes between are free, otherwise the route slips
            # through the gap between two obstacle cells.
            if dr and dc and (grid[r + dr, c] == BLOCKED or grid[r, c + dc] == BLOCKED):
                continue

            tentative = g + step * float(cost[nb])
            if tentative < g_score.get(nb, float("inf")):
                g_score[nb] = tentative
                came_from[nb] = current
                heapq.heappush(open_heap, (tentative + heuristic(nb, goal), tentative, nb))

    return []


def nearest_free(grid, rc, radius=12):
    """
    Snap a point onto the free space around it.

    The operator clicks a casualty that segmentation called non-traversable
    (they are lying on rubble — that is the whole scenario), and the planner
    would report "unreachable" for a goal that is plainly right there. Expand
    a ring until a free cell turns up.
    """
    h, w = grid.shape
    r0, c0 = rc
    if 0 <= r0 < h and 0 <= c0 < w and grid[r0, c0] == FREE:
        return rc

    for rad in range(1, radius + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                r, c = r0 + dr, c0 + dc
                if 0 <= r < h and 0 <= c < w and grid[r, c] == FREE:
                    return (r, c)
    return None


def simplify(path, tolerance=1.0):
    """Collapse collinear runs so the overlay draws a few segments, not 400."""
    if len(path) < 3:
        return path
    pts = np.array(path, dtype=np.float32).reshape(-1, 1, 2)
    return [tuple(map(int, p[0])) for p in cv2.approxPolyDP(pts, tolerance, False)]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class SegWorker(BaseWorker):
    NAME = "seg"

    def __init__(self, *args, backend="stub", checkpoint=None, grid=64,
                 clearance=3.0, start=None, goal=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.checkpoint = checkpoint
        self.grid_w = grid
        self.clearance = clearance          # cost falls off over this many cells
        self.start_rc = start
        self.goal_rc = goal
        self._model = None
        self._device = "cpu"
        self._last_shape = None

    # -- load --------------------------------------------------------------

    def load(self):
        if self.backend == "stub":
            log.warning("running the STUB segmentation backend — not a trained model")
            return
        self._load_segformer()

    def _load_segformer(self):
        """
        Real backend for SegFormer/segformer_best.pth.

        The checkpoint in the repo is a state_dict, so the architecture has to
        be constructed first. Adjust num_labels/repo to match how it was
        trained (Programs/Python/SegFormer/training.py) — if the head shape in
        the checkpoint disagrees, load_state_dict says so loudly rather than
        silently producing noise.
        """
        import torch
        from transformers import SegformerForSemanticSegmentation

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = self.checkpoint or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "SegFormer", "segformer_best.pth"
        )
        ckpt = os.path.abspath(ckpt)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"segformer checkpoint not found: {ckpt}")

        log.info("loading SegFormer from %s on %s", ckpt, self._device)
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        model.to(self._device).eval()

        self._model = model
        self._torch = torch
        log.info("segmentation model ready")

    # -- commands ----------------------------------------------------------

    def on_command(self, payload):
        """
        plan_path { start: [row, col], goal: [row, col] }

        Coordinates are grid cells, which is what the pane sends: it renders
        the grid at its natural resolution and maps a click straight to a cell,
        so no scaling assumption is shared between the two sides.
        """
        if payload.get("start") is not None:
            self.start_rc = tuple(int(v) for v in payload["start"])
        if payload.get("goal") is not None:
            self.goal_rc = tuple(int(v) for v in payload["goal"])
        if payload.get("clearance") is not None:
            self.clearance = float(payload["clearance"])
        if payload.get("action") == "clear_path":
            self.start_rc = self.goal_rc = None
        log.info("plan target: start=%s goal=%s", self.start_rc, self.goal_rc)

    # -- segmentation ------------------------------------------------------

    def _segment(self, frame):
        """Return a uint8 mask, 255 = traversable."""
        return (self._seg_stub(frame) if self.backend == "stub"
                else self._seg_segformer(frame))

    def _seg_segformer(self, frame):
        torch = self._torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = ((rgb - mean) / std).transpose(2, 0, 1)[None]

        with torch.no_grad():
            logits = self._model(pixel_values=torch.from_numpy(x).to(self._device)).logits

        # SegFormer emits logits at 1/4 input resolution.
        logits = torch.nn.functional.interpolate(
            logits, size=frame.shape[:2], mode="bilinear", align_corners=False
        )
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        return np.where(pred == 0, 255, 0).astype(np.uint8)

    @staticmethod
    def _seg_stub(frame):
        """
        Placeholder traversability. Open ground in an aerial frame tends to be
        low-texture and mid-brightness; rubble, vegetation and structures are
        high-texture or extreme. Threshold on that, then morphologically close
        so the mask is made of regions a planner can actually route through
        rather than speckle.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        texture = cv2.GaussianBlur(np.abs(texture), (0, 0), 5)
        rough = texture > np.percentile(texture, 72)

        # Flat dark regions are structures (a roof from above has almost no
        # texture, so the texture term alone passes straight over a building
        # and calls it open ground); blown-out regions are usually specular or
        # sky. The dark cutoff is deliberately generous — under-calling
        # traversable ground routes a responder the long way round, which is
        # recoverable, while calling a building traversable routes them into a
        # wall.
        v = gray.astype(np.float32) / 255.0
        extreme = (v < 0.30) | (v > 0.92)

        blocked = rough | extreme
        mask = np.where(blocked, 0, 255).astype(np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # -- grid + cost -------------------------------------------------------

    def _build_grid(self, mask):
        h, w = mask.shape
        gw = self.grid_w
        gh = max(4, int(round(gw * h / w)))

        # INTER_AREA averages the cell, so a cell is free only if most of it is
        # free — nearest-neighbour would let a single free pixel open a route
        # through a wall.
        small = cv2.resize(mask, (gw, gh), interpolation=cv2.INTER_AREA)
        grid = np.where(small > 127, FREE, BLOCKED).astype(np.uint8)

        # Distance from each free cell to the nearest blocked cell. Cost decays
        # from 1 + clearance at the edge of an obstacle to 1.0 well clear of
        # one, so A* is nudged toward the middle of open ground without ever
        # being told a free cell is impassable.
        free_u8 = (grid == FREE).astype(np.uint8)
        dist = cv2.distanceTransform(free_u8, cv2.DIST_L2, 3)
        proximity = np.clip(1.0 - dist / max(1e-6, self.clearance), 0.0, 1.0)
        cost = (1.0 + self.clearance * proximity).astype(np.float32)

        return grid, cost

    @staticmethod
    def _largest_region(grid):
        """Mask of the biggest connected run of free cells, or None if there is none."""
        count, labels = cv2.connectedComponents((grid == FREE).astype(np.uint8))
        if count <= 1:
            return None
        # Label 0 is the blocked background; pick the largest of the rest.
        sizes = [(int((labels == i).sum()), i) for i in range(1, count)]
        return labels == max(sizes)[1]

    def _default_endpoints(self, grid):
        """
        With nothing selected in the UI, plan something sensible so the pane is
        never blank: from where a responder enters frame (bottom-centre) to the
        furthest point from there.

        Both ends are constrained to the LARGEST connected open region. Picking
        them from the free cells at large routinely straddles two regions — a
        speck of open ground inside the rubble is "furthest from the start" by
        Euclidean distance while being unreachable — and the pane then reports
        no route on a frame that plainly has one.
        """
        region = self._largest_region(grid)
        if region is None:
            return None, None

        cells = np.argwhere(region)
        h, w = grid.shape

        # Start: the region cell nearest the responder's entry point.
        entry = np.array([h - 1, w // 2], dtype=np.float32)
        start = tuple(int(v) for v in cells[int(np.argmin(
            np.hypot(cells[:, 0] - entry[0], cells[:, 1] - entry[1])
        ))])

        goal = tuple(int(v) for v in cells[int(np.argmax(
            np.hypot(cells[:, 0] - start[0], cells[:, 1] - start[1])
        ))])
        return start, goal

    # -- render ------------------------------------------------------------

    def _render(self, frame, mask, grid, path, start, goal):
        vis = frame.copy()
        h, w = frame.shape[:2]

        # Non-traversable in red, traversable in green, both at low alpha so
        # the operator still sees the scene underneath.
        overlay = np.zeros_like(frame)
        overlay[mask > 127] = (60, 200, 60)
        overlay[mask <= 127] = (40, 40, 220)
        vis = cv2.addWeighted(vis, 0.68, overlay, 0.32, 0)

        gh, gw = grid.shape
        sx, sy = w / gw, h / gh

        def to_px(rc):
            return int((rc[1] + 0.5) * sx), int((rc[0] + 0.5) * sy)

        if len(path) > 1:
            pts = np.array([to_px(p) for p in path], dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.polylines(vis, [pts], False, (80, 220, 255), 3, cv2.LINE_AA)

        if start:
            cv2.circle(vis, to_px(start), 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, to_px(start), 7, (255, 200, 60), -1, cv2.LINE_AA)
        if goal:
            gx, gy = to_px(goal)
            cv2.drawMarker(vis, (gx, gy), (0, 0, 0), cv2.MARKER_CROSS, 26, 7, cv2.LINE_AA)
            cv2.drawMarker(vis, (gx, gy), (80, 80, 255), cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA)

        tag = "TRAVERSABILITY · STUB" if self.backend == "stub" else f"TRAVERSABILITY · SegFormer · {self._device}"
        label_frame(vis, f"{tag}   grid {gw}x{gh}   {'route ' + str(len(path)) + ' cells' if path else 'no route'}",
                    (140, 255, 180))
        return vis

    # -- infer -------------------------------------------------------------

    def infer(self, frame):
        mask = self._segment(frame)
        grid, cost = self._build_grid(mask)

        start = nearest_free(grid, self.start_rc) if self.start_rc else None
        goal = nearest_free(grid, self.goal_rc) if self.goal_rc else None

        if start is None or goal is None:
            auto_start, auto_goal = self._default_endpoints(grid)
            start = start or auto_start
            goal = goal or auto_goal

        path = astar(grid, cost, start, goal) if (start and goal) else []

        vis = self._render(frame, mask, grid, path, start, goal)

        free_frac = float((grid == FREE).mean())
        length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])
        )

        return vis, {
            "backend": self.backend,
            "device": self._device,
            "grid": grid.tolist(),          # the pane draws its own copy for hit-testing
            "gridSize": [int(grid.shape[1]), int(grid.shape[0])],
            "path": [[int(r), int(c)] for r, c in simplify(path)],
            "pathCells": len(path),
            "pathLengthCells": round(length, 2),
            "start": list(start) if start else None,
            "goal": list(goal) if goal else None,
            "traversableFraction": round(free_frac, 4),
            "reachable": bool(path),
        }


def main():
    def rc(value):
        r, c = value.split(",")
        return (int(r), int(c))

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, image, camera index, or MJPEG URL")
    ap.add_argument("--backend", choices=["stub", "segformer"], default="stub")
    ap.add_argument("--checkpoint", default=None, help="segformer .pth (default: ../SegFormer/segformer_best.pth)")
    ap.add_argument("--grid", type=int, default=64, help="grid width in cells")
    ap.add_argument("--clearance", type=float, default=3.0, help="obstacle-avoidance weight, in cells")
    ap.add_argument("--start", type=rc, default=None, metavar="ROW,COL")
    ap.add_argument("--goal", type=rc, default=None, metavar="ROW,COL")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--url", default=os.environ.get("GUI_URL", "http://127.0.0.1:7100"))
    args = ap.parse_args()

    SegWorker(
        args.source, url=args.url, fps=args.fps, loop=not args.no_loop, width=args.width,
        backend=args.backend, checkpoint=args.checkpoint, grid=args.grid,
        clearance=args.clearance, start=args.start, goal=args.goal,
    ).start()


if __name__ == "__main__":
    main()

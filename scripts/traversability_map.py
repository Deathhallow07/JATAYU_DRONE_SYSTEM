"""
Traversability map from an ODM orthomosaic, for global path planning.

Design note -- why this is not "YOLOE with better prompts":

  YOLOE is an open-vocabulary *instance* segmentation model trained on
  LVIS / Objects365 / GoldG. It segments "things" (countable objects with
  a meaningful bounding box). Ground cover -- grass, dirt, track, gravel --
  is "stuff": no instances, no meaningful box. Measured on this ortho,
  the prompts "grass"/"vegetation"/"lawn"/"dirt road"/"pavement" return
  nothing above conf 0.05, and at conf 0.01 the only survivor is a single
  0.03-confidence blob covering 85% of the frame. That is noise, not
  segmentation. No prompt rewording fixes a thing/stuff mismatch.

  So the labour is split by what each tool is actually good at:
    surface type  -> Lab colour clustering (grass vs bare earth vs paving)
    obstacles     -> YOLOE (person scored 0.86 here; that is its strength)

  Output is a COST grid, not a binary mask. For a rover on an open field
  almost everything is drivable; what matters is *preference* (packed
  track < grass) plus hard obstacle exclusion with a safety margin.

!! SAFETY LIMIT -- READ BEFORE REUSING THIS ON ANOTHER SITE !!

  This script assumes "not detected => traversable". That assumption is
  ONLY defensible here because this ortho is ~99% open ground, verified by
  eye. It does not generalise, and it fails silently.

  Measured on dtu_tour.mp4 (campus: buildings, canopy, scaffolding), with a
  22-prompt vocabulary covering building/tree/wall/roof/crane/container/...:

      frame        claimed    would be called TRAVERSABLE
      tour_0          0.0%                         100.0%
      tour_1          2.9%                          97.1%
      tour_2          0.0%                         100.0%
      tour_3          0.1%                          99.9%
      tour_4          1.3%                          98.7%

  Prompting for "tree" alone returned 0 instances on all 5 frames, one of
  which is ~40% tree canopy. Separately, dropping "person" from the ortho
  vocabulary made all 10 people vanish into "traversable".

  A detector's failure mode is silence; negation reads silence as "safe".
  You cannot enumerate a vocabulary of everything that blocks a rover --
  that is the open-set problem, and no prompt list closes it.

  For anything busier than an open field use dense semantic segmentation
  (dinov2_traversability.py), where every pixel is assigned a class. Such a
  model can be WRONG, but it is never SILENT -- and geometry (an ODM DSM,
  height-above-ground) is stronger still, because it does not depend on
  recognising the obstacle at all.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

# =====================================================
# CONFIG
# =====================================================

ORTHO_PATH = "/media/uasdtu/DataSets2/Segmentation_GATE_1/odm_orthophoto.png"
MODEL_PATH = "/media/uasdtu/DataSets2/Segmentation_GATE_1/yoloe-26x-seg.pt"
OUT_PREFIX = "/media/uasdtu/DataSets2/Segmentation_GATE_1/traversability/traversability"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Ground sampling distance of the ortho, metres/pixel. READ THIS FROM YOUR
# ODM run (odm_orthophoto/odm_orthophoto_log.txt, or the geotiff transform).
# Everything below in metres depends on it being right.
GSD_M_PER_PX = 0.03

ROVER_RADIUS_M = 0.40      # half the widest dimension of the vehicle
SAFETY_MARGIN_M = 0.30     # extra standoff around every obstacle
PLANNER_CELL_M = 0.25      # output grid resolution

OBSTACLE_PROMPTS = [
    "person",
    "car",
    "tree", 
    "bench", 
    "pole", 
    "building", 
    "bush", 
    "fence",
]
OBSTACLE_CONF = 0.25

# Tile size for YOLOE. The ortho is far larger than the model's input;
# feeding it whole downscales a person to a handful of pixels.
TILE = 1024
TILE_OVERLAP = 128

# Relative driving cost per surface. 1.0 = ideal.
COST_TRACK = 1.0     # packed bare earth / the track
COST_PAVING = 1.1    # the paved area
COST_GRASS = 1.6     # higher rolling resistance, hides ruts
COST_BLOCKED = np.inf


# =====================================================
# 1. LOAD ORTHO + VALID (STITCHED) FOOTPRINT
# =====================================================

def load_ortho(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Could not open ortho: {path}")

    if im.shape[2] == 4:
        bgr, valid = im[:, :, :3], im[:, :, 3] > 0
    else:
        bgr = im
        # ODM pads with pure black outside the stitch footprint
        valid = im.any(axis=2)

    return bgr, valid


# =====================================================
# 2. SURFACE TYPE -- Lab COLOUR CLUSTERING
# =====================================================

def segment_surface(bgr, valid, k=4):
    """Cluster ground cover in Lab. Returns a label image + per-cluster stats.

    Lab is used rather than RGB/HSV because a* separates vegetation from
    bare earth almost linearly (chlorophyll -> negative a*), and it stays
    stable under the brightness variation that orthomosaic blending leaves
    between passes.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    greenness = 128.0 - a          # higher = greener
    yellowness = b - 128.0

    # L is downweighted so shadows don't dominate the clustering
    feat = np.stack(
        [greenness[valid], yellowness[valid], L[valid] * 0.5], axis=1
    ).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(
        feat, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    seg = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    seg[valid] = labels.ravel()

    # Rank clusters by greenness: greenest is grass, brightest of the rest
    # is paving, remainder is bare earth / track.
    by_green = np.argsort(-centers[:, 0])
    grass_ids = {int(by_green[0])}

    rest = [int(c) for c in by_green[1:]]
    paving_id = max(rest, key=lambda c: centers[c, 2])

    return seg, grass_ids, {paving_id}, set(rest) - {paving_id}


def clean_mask(mask, open_px=5, close_px=9):
    """Despeckle. k-means is per-pixel and ignores spatial structure, so raw
    output is salt-and-pepper; a planner wants connected regions."""
    m = mask.astype(np.uint8)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    return m.astype(bool)


# =====================================================
# 3. OBSTACLES -- TILED YOLOE
# =====================================================

def detect_obstacles(bgr, valid):
    """Run YOLOE over overlapping tiles, union the masks of every hit."""
    print("Loading YOLOE...")
    model = YOLOE(MODEL_PATH)
    model.to(DEVICE)
    model.set_classes(OBSTACLE_PROMPTS, model.get_text_pe(OBSTACLE_PROMPTS))

    H, W = bgr.shape[:2]
    obstacles = np.zeros((H, W), dtype=bool)
    hits = []

    step = TILE - TILE_OVERLAP
    ys = list(range(0, max(H - TILE, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - TILE, 0) + 1, step)) or [0]
    if ys[-1] + TILE < H:
        ys.append(H - TILE)
    if xs[-1] + TILE < W:
        xs.append(W - TILE)

    print(f"Scanning {len(ys) * len(xs)} tiles of {TILE}px...")

    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + TILE, H), min(x0 + TILE, W)
            tile = bgr[y0:y1, x0:x1]

            # Skip tiles that are mostly outside the stitch footprint
            if valid[y0:y1, x0:x1].mean() < 0.10:
                continue

            with torch.no_grad():
                r = model.predict(
                    tile, conf=OBSTACLE_CONF, device=DEVICE,
                    verbose=False, retina_masks=True,
                )[0]

            if r.masks is None or len(r.boxes) == 0:
                continue

            masks = r.masks.data.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cfs = r.boxes.conf.cpu().numpy()

            for m, c, cf in zip(masks, cls, cfs):
                m = cv2.resize(
                    m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
                ) > 0.5

                # A tile-filling "detection" is the degenerate whole-image
                # blob, not an object. Reject it.
                if m.mean() > 0.50:
                    continue

                obstacles[y0:y1, x0:x1] |= m
                hits.append((OBSTACLE_PROMPTS[c], float(cf)))

    obstacles &= valid
    return obstacles, hits


# =====================================================
# 4. COST GRID
# =====================================================

def build_cost(valid, grass, paving, track, obstacles):
    H, W = valid.shape
    cost = np.full((H, W), COST_BLOCKED, dtype=np.float32)

    cost[track] = COST_TRACK
    cost[paving] = COST_PAVING
    cost[grass] = COST_GRASS

    # Unmapped area is not known-safe, so it stays blocked.
    cost[~valid] = COST_BLOCKED

    # Inflate obstacles by rover radius + margin, so the planner can treat
    # the rover as a point.
    inflate_px = int(round((ROVER_RADIUS_M + SAFETY_MARGIN_M) / GSD_M_PER_PX))
    if inflate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * inflate_px + 1, 2 * inflate_px + 1)
        )
        inflated = cv2.dilate(obstacles.astype(np.uint8), k).astype(bool)
    else:
        inflated = obstacles

    cost[inflated] = COST_BLOCKED
    return cost, inflated


def downsample_cost(cost):
    """Reduce to planner resolution. Max-pool the cost so a blocked cell
    anywhere inside a planner cell blocks the whole cell -- conservative,
    which is what you want for a vehicle."""
    factor = max(1, int(round(PLANNER_CELL_M / GSD_M_PER_PX)))
    H, W = cost.shape
    Hc, Wc = H // factor, W // factor

    trimmed = cost[:Hc * factor, :Wc * factor]
    blocks = trimmed.reshape(Hc, factor, Wc, factor)
    return blocks.max(axis=(1, 3)), factor


# =====================================================
# MAIN
# =====================================================

def main():
    bgr, valid = load_ortho(ORTHO_PATH)
    H, W = bgr.shape[:2]
    print(f"Ortho {W}x{H}, {100 * valid.mean():.1f}% inside stitch footprint")
    print(f"GSD {GSD_M_PER_PX} m/px -> {W * GSD_M_PER_PX:.0f}m x {H * GSD_M_PER_PX:.0f}m\n")

    seg, grass_ids, paving_ids, _ = segment_surface(bgr, valid)

    grass = clean_mask(np.isin(seg, list(grass_ids)) & valid)
    paving = clean_mask(np.isin(seg, list(paving_ids)) & valid)
    track = valid & ~grass & ~paving

    for name, m in [("grass", grass), ("paving", paving), ("track/bare", track)]:
        print(f"  {name:<12} {100 * m.sum() / valid.sum():5.1f}% of mapped area")

    obstacles, hits = detect_obstacles(bgr, valid)
    print(f"\n{len(hits)} obstacle instances, {100 * obstacles.sum() / valid.sum():.2f}% of area")
    from collections import Counter
    for cname, n in Counter(h[0] for h in hits).most_common():
        print(f"  {cname:<12} n={n}")

    cost, inflated = build_cost(valid, grass, paving, track, obstacles)
    grid, factor = downsample_cost(cost)

    drivable = np.isfinite(cost) & valid
    print(f"\nDrivable after inflation: {100 * drivable.sum() / valid.sum():.1f}% of mapped area")
    print(f"Planner grid {grid.shape[1]}x{grid.shape[0]} @ {PLANNER_CELL_M}m "
          f"({factor}px/cell), {100 * np.isfinite(grid).mean():.1f}% open")

    np.save(f"{OUT_PREFIX}_cost_grid.npy", grid)

    # ---- visualisation ----
    vis = bgr.copy()

    def blend(mask, colour, alpha=0.5):
        vis[mask] = (
            (1 - alpha) * vis[mask] + alpha * np.array(colour, np.float32)
        ).astype(np.uint8)

    blend(track, (60, 200, 60))       # green = best surface
    blend(paving, (0, 220, 220))      # yellow = ok
    blend(grass, (200, 150, 0))       # blue   = costlier
    blend(inflated & ~obstacles, (0, 100, 255), 0.55)   # orange = margin
    blend(obstacles, (0, 0, 255), 0.85)                 # red    = obstacle
    vis[~valid] = 0

    cv2.imwrite(f"{OUT_PREFIX}_overlay.png", vis)

    hard = np.zeros((H, W, 3), np.uint8)
    hard[drivable] = (255, 255, 255)
    cv2.imwrite(f"{OUT_PREFIX}_binary.png", hard)

    print(f"\nSaved:\n  {OUT_PREFIX}_overlay.png"
          f"\n  {OUT_PREFIX}_binary.png"
          f"\n  {OUT_PREFIX}_cost_grid.npy  (inf = blocked)")


if __name__ == "__main__":
    main()

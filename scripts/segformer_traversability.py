"""
Traversability from the UAVid-trained SegFormer-B0, tiled over the ortho.

This is the semantic-segmentation counterpart to traversability_map.py.
SegFormer predicts a class for EVERY pixel (dense "stuff" segmentation),
which is the task YOLOE structurally cannot do.

UAVid class -> traversability mapping is declared in TRAVERSABILITY below.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

ORTHO_PATH = "/media/uasdtu/DataSets2/Segmentation_GATE_1/odm_orthophoto.png"
MODEL_PATH = "/media/uasdtu/DataSets2/Segmentation_GATE_1/segformer_b0_best_updated"
OUT_PREFIX = "/media/uasdtu/DataSets2/Segmentation_GATE_1/traversability/segformer_trav"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# The 8 classes the checkpoint was actually trained on, in index order.
# NOTE: inference.py lists only 5 names -- that list is desynced from the
# checkpoint and mislabels every class. This is the correct order.
CLASS_NAMES = [
    "Clutter", "Building", "Road", "Static_Car",
    "Tree", "Vegetation", "Human", "Moving_Car",
]

# Per-class driving cost. inf = hard obstacle.
# Validation IoU from training_log.csv is noted, because a class the model
# cannot actually predict should not be trusted in a planner.
TRAVERSABILITY = {
    "Clutter":    2.0,       # iou 0.56 - catch-all, treat as passable-but-avoid
    "Building":   np.inf,    # iou 0.88
    "Road":       1.0,       # iou 0.71 - best surface
    "Static_Car": np.inf,    # iou 0.40
    "Tree":       np.inf,    # iou 0.73
    "Vegetation": 1.6,       # iou 0.65 - grass/low veg, drivable, costlier
    "Human":      np.inf,    # iou 0.17 - UNRELIABLE, see notes
    "Moving_Car": np.inf,    # iou 0.51
}

TILE = 1024
OVERLAP = 256


def load_ortho(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Could not open ortho: {path}")
    if im.shape[2] == 4:
        return im[:, :, :3], im[:, :, 3] > 0
    return im, im.any(axis=2)


def predict_tiled(bgr, valid):
    """Sliding-window inference, accumulating per-class logits.

    Overlapping tiles are averaged rather than overwritten, which removes
    the seam artefacts you get from hard tile boundaries.
    """
    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
    model.to(DEVICE).eval()

    H, W = bgr.shape[:2]
    n_cls = len(CLASS_NAMES)
    acc = np.zeros((n_cls, H, W), dtype=np.float32)
    hits = np.zeros((H, W), dtype=np.float32)

    step = TILE - OVERLAP
    ys = list(range(0, max(H - TILE, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - TILE, 0) + 1, step)) or [0]
    if ys[-1] + TILE < H:
        ys.append(H - TILE)
    if xs[-1] + TILE < W:
        xs.append(W - TILE)

    print(f"SegFormer over {len(ys) * len(xs)} tiles...")

    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + TILE, H), min(x0 + TILE, W)
            if valid[y0:y1, x0:x1].mean() < 0.10:
                continue

            rgb = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            inp = processor(images=rgb, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                logits = model(**inp).logits

            logits = F.interpolate(
                logits, size=(y1 - y0, x1 - x0),
                mode="bilinear", align_corners=False,
            )
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()

            acc[:, y0:y1, x0:x1] += prob
            hits[y0:y1, x0:x1] += 1.0

    hits[hits == 0] = 1.0
    return (acc / hits).argmax(0).astype(np.uint8)


def main():
    bgr, valid = load_ortho(ORTHO_PATH)
    pred = predict_tiled(bgr, valid)

    print("\nClass distribution inside stitch footprint:")
    total = valid.sum()
    counts = Counter(pred[valid].ravel().tolist())
    for cid, n in counts.most_common():
        print(f"  {CLASS_NAMES[cid]:<12} {100 * n / total:5.1f}%   cost={TRAVERSABILITY[CLASS_NAMES[cid]]}")

    cost = np.full(pred.shape, np.inf, dtype=np.float32)
    for cid, name in enumerate(CLASS_NAMES):
        cost[(pred == cid) & valid] = TRAVERSABILITY[name]
    cost[~valid] = np.inf

    drivable = np.isfinite(cost)
    print(f"\nDrivable: {100 * drivable.sum() / total:.1f}% of mapped area")

    np.save(f"{OUT_PREFIX}_cost.npy", cost)

    # UAVid palette, for eyeballing what the model actually thinks
    palette = np.array([
        (0, 0, 0), (0, 0, 128), (128, 64, 128), (192, 0, 192),
        (0, 128, 0), (0, 128, 128), (0, 64, 64), (128, 0, 64),
    ], dtype=np.uint8)

    seg_vis = palette[pred]
    seg_vis[~valid] = 0
    blend = (0.45 * bgr + 0.55 * seg_vis).astype(np.uint8)
    blend[~valid] = 0
    cv2.imwrite(f"{OUT_PREFIX}_seg.png", blend)

    hard = np.zeros(bgr.shape, np.uint8)
    hard[drivable] = (255, 255, 255)
    cv2.imwrite(f"{OUT_PREFIX}_binary.png", hard)
    print(f"Saved {OUT_PREFIX}_seg.png / _binary.png / _cost.npy")


if __name__ == "__main__":
    main()

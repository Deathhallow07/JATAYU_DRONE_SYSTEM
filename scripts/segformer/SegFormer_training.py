"""
Navigation-oriented multi-dataset semantic segmentation for UAV -> rover global planning.

Single SegFormer (MiT-B3) trained on a unified 14-class navigation ontology, fed by
UAVid / LoveDA / OpenEarthMap (and later our own field data), each remapped into that common label
space. Training runs as progressive stages: OpenEarthMap -> LoveDA -> UAVid -> field,
each stage initialized from the previous stage's best checkpoint.

Usage
-----
  python SegFormer_training.py --check-data            # verify discovery + label stats, no training
  python SegFormer_training.py                         # run all configured stages in order
  python SegFormer_training.py --stages uavid,field    # run a subset
  python SegFormer_training.py --stages field --init-from ./segformer_b3_multidataset/stage3_uavid
"""

import os
import csv
import glob
import json
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None  # UAVid frames are 4096x2160; disable the decompression-bomb guard

# ==========================================
# 0. USER CONFIGURATION
# ==========================================
DATA_DIR = "./Datasets"
OUTPUT_DIR = "./segformer_b3_multidataset"
PRETRAINED_CKPT = "nvidia/mit-b3"   # ImageNet-pretrained encoder only — NOT an ADE20K/Cityscapes seg checkpoint

SEED = 42
BATCH_SIZE = 8
CROP_SIZE = 512                     # train on random crops; val on resized full frames
VAL_SIZE = 768                      # long-side resize for validation
LR_HEAD = 6e-5                      # decode head (randomly init'd) — learns faster
LR_BACKBONE = 6e-6                  # pretrained encoder — smaller LR to avoid catastrophic forgetting
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.05
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4
CLASS_WEIGHT_SAMPLES = 400          # masks sampled per stage to estimate class frequencies

# The ground-surface classes the traversability cost map is built from. Getting these right
# IS the task — a rover plans over the drivable surface, so confusing Grass with Soil or Road
# with Pavement changes the path, while missing a parked car barely moves the cost field.
# Only classes actually present in a stage's val set contribute (LoveDA has no Pavement, etc.).
CRITICAL_CLASSES = ["Road", "Pavement", "Soil", "Grass", "Vegetation"]

# ==========================================
# 1. COMMON NAVIGATION ONTOLOGY
# ==========================================
# id -> (name, RGB for visualization). Class 0 is the Ignore/Unknown slot required by the
# ontology spec; unlabeled pixels are written as IGNORE_INDEX (255) so they contribute to
# neither loss nor metrics, and class 0 is excluded from the mIoU average.
COMMON_COLORMAP = {
    0:  ("Ignore",     (0, 0, 0)),
    1:  ("Road",       (128, 64, 128)),
    2:  ("Pavement",   (244, 35, 232)),
    3:  ("Soil",       (152, 102, 52)),
    4:  ("Grass",      (152, 251, 152)),
    5:  ("Tree",       (0, 102, 0)),
    6:  ("Vegetation", (154, 205, 50)),
    7:  ("Building",   (128, 0, 0)),
    8:  ("Vehicle",    (0, 0, 142)),
    9:  ("Person",     (220, 20, 60)),
    10: ("Pole",       (153, 153, 153)),
    11: ("Wall",       (102, 102, 156)),
    12: ("Water",      (0, 130, 180)),
    13: ("Rubble",     (150, 100, 100)),
}

NUM_CLASSES = len(COMMON_COLORMAP)
CLASS_NAMES = [COMMON_COLORMAP[i][0] for i in range(NUM_CLASSES)]
CLASS_COLORS = np.array([COMMON_COLORMAP[i][1] for i in range(NUM_CLASSES)], dtype=np.uint8)
IGNORE_INDEX = 255
UNUSED_CLASSES = [0]  # excluded from mIoU / selection metric

# Traversability grouping — the binary decision the global planner actually consumes.
# Class 0 (Ignore/Unknown) sits on the NON-traversable side deliberately: an unknown
# prediction must never be planned through.
TRAVERSABLE_IDS = [1, 2, 3, 4, 6]                    # Road, Pavement, Soil, Grass, Low Vegetation
NON_TRAVERSABLE_IDS = [0, 5, 7, 8, 9, 10, 11, 12, 13]  # Unknown, Tree, Building, Vehicle, Person,
                                                       # Pole, Wall, Water, Rubble

# ==========================================
# 2. PER-DATASET REMAPPING INTO THE COMMON ONTOLOGY
# ==========================================
# UAVid ships RGB-coded masks.
UAVID_RGB_TO_COMMON = {
    (0, 0, 0):       IGNORE_INDEX,  # "background clutter" is a catch-all (sky, water, rubble...) — not a usable class
    (128, 0, 0):     7,             # Building
    (128, 64, 128):  1,             # Road
    (0, 128, 0):     5,             # Tree
    (128, 128, 0):   6,             # Low vegetation
    (64, 0, 128):    8,             # Moving car   -> Vehicle
    (192, 0, 192):   8,             # Static car   -> Vehicle
    (64, 64, 0):     9,             # Human        -> Person
}

# LoveDA ships single-channel label-id masks (0 = no-data).
LOVEDA_ID_TO_COMMON = {
    0: IGNORE_INDEX,  # no-data
    1: IGNORE_INDEX,  # "background" is an undifferentiated mixture — ignoring beats mislabeling
    2: 7,             # building
    3: 1,             # road
    4: 12,            # water
    5: 3,             # barren      -> Soil
    6: 5,             # forest      -> Tree
    7: 6,             # agriculture -> Low Vegetation (cropland, not lawn)
}

# OpenEarthMap ships single-channel label-id masks (0 = unknown).
OEM_ID_TO_COMMON = {
    0: IGNORE_INDEX,  # unknown
    1: 3,             # bareland        -> Soil
    2: 4,             # rangeland       -> Grass
    3: 2,             # developed space -> Pavement
    4: 1,             # road
    5: 5,             # tree
    6: 12,            # water
    7: 6,             # agriculture     -> Low Vegetation
    8: 7,             # building
}

# Field masks are authored directly in the common ontology; 0 stays the Ignore slot.
FIELD_ID_TO_COMMON = {0: IGNORE_INDEX, **{i: i for i in range(1, NUM_CLASSES)}}


def _lut_from_dict(mapping, size=256):
    """Build a 256-entry uint8 LUT so id remapping is a single vectorized gather."""
    lut = np.full(size, IGNORE_INDEX, dtype=np.uint8)
    for src, dst in mapping.items():
        lut[src] = dst
    return lut


LOVEDA_LUT = _lut_from_dict(LOVEDA_ID_TO_COMMON)
OEM_LUT = _lut_from_dict(OEM_ID_TO_COMMON)
FIELD_LUT = _lut_from_dict(FIELD_ID_TO_COMMON)


def _load_label_array(path):
    """Read a mask without corrupting it.

    Paletted PNGs must NOT go through .convert("L") — that would replace class indices
    with luminance values. np.array() on a "P" image returns the raw indices, which is
    what we want. RGB masks are returned as HxWx3.
    """
    img = Image.open(path)
    if img.mode in ("P", "L", "I", "I;16"):
        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.int32)
        return arr
    return np.array(img.convert("RGB"))


def remap_uavid(mask) -> np.ndarray:
    if mask.ndim == 2:  # some UAVid mirrors distribute paletted masks
        raise ValueError("UAVid masks are expected to be RGB-coded; got a single-channel mask.")
    out = np.full(mask.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    # Pack RGB into a single int32 so each class is one comparison instead of three.
    packed = (mask[..., 0].astype(np.int32) << 16) | (mask[..., 1].astype(np.int32) << 8) | mask[..., 2]
    for rgb, cid in UAVID_RGB_TO_COMMON.items():
        key = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
        out[packed == key] = cid
    return out


def _remap_ids(mask, lut, name):
    if mask.ndim == 3:
        raise ValueError(f"{name} masks are expected to be single-channel label ids; got RGB.")
    arr = np.asarray(mask)
    if arr.max() >= len(lut):
        raise ValueError(f"{name} mask contains id {arr.max()} outside the expected range.")
    return lut[arr.astype(np.uint8)]


def remap_loveda(mask) -> np.ndarray:
    return _remap_ids(mask, LOVEDA_LUT, "LoveDA")


def remap_openearthmap(mask) -> np.ndarray:
    return _remap_ids(mask, OEM_LUT, "OpenEarthMap")


def remap_field(mask) -> np.ndarray:
    return _remap_ids(mask, FIELD_LUT, "FIELD")


REMAP_FNS = {
    "uavid": remap_uavid,
    "loveda": remap_loveda,
    "openearthmap": remap_openearthmap,
    "field": remap_field,
}

# ==========================================
# 3. DATASET DISCOVERY
# ==========================================
# Each dataset keeps its own image/label directory naming. Pairs are matched by file stem
# so a .tif image can pair with a .tif label and a .png image with a .png label.
DATASET_LAYOUTS = {
    "uavid": {
        "root_names": ["UAVid", "uavid"],
        "img_dirs": ["Images", "images"],
        "lbl_dirs": ["Labels", "labels", "TrainId", "Annotations"],
        "train_markers": ["uavid_train"],
        "val_markers": ["uavid_val"],
    },
    "loveda": {
        "root_names": ["LoveDA", "loveda"],
        "img_dirs": ["images_png", "images"],
        "lbl_dirs": ["masks_png", "masks", "labels"],
        "train_markers": ["train"],
        "val_markers": ["val"],
    },
    "openearthmap": {
        "root_names": ["OpenEarthMap", "openearthmap", "OEM"],
        "img_dirs": ["images"],
        "lbl_dirs": ["label", "labels"],
        "train_markers": ["train"],
        "val_markers": ["val"],
    },
    "field": {
        "root_names": ["FIELD", "field"],
        "img_dirs": ["images", "Images"],
        "lbl_dirs": ["labels", "Labels", "masks"],
        "train_markers": ["train"],
        "val_markers": ["val"],
    },
}


def find_dataset_root(data_root, names):
    for name in names:
        direct = os.path.join(data_root, name)
        if os.path.isdir(direct):
            return direct
    for root, dirs, _ in os.walk(data_root):
        for d in dirs:
            if d in names:
                return os.path.join(root, d)
    return None


def _matching_label_dir(dataset_root, img_dir, layout):
    """Find the label directory corresponding to an image directory.

    Works by substituting the image-directory component of the path with each candidate
    label name, which covers every layout in use here:
      UAVid   seq1/Images      -> seq1/Labels
      LoveDA  Urban/images_png -> Urban/masks_png
      OEM     images/train     -> label/train      (split nested *under* the image dir)
    """
    rel = os.path.relpath(img_dir, dataset_root).split(os.sep)
    for i in range(len(rel) - 1, -1, -1):
        if rel[i] not in layout["img_dirs"]:
            continue
        for name in layout["lbl_dirs"]:
            cand = os.path.join(dataset_root, *rel[:i], name, *rel[i + 1:])
            if os.path.isdir(cand):
                return cand
    return None


def collect_pairs(dataset_root, layout):
    """Walk the dataset and return every (image_path, label_path) pair, matched by stem."""
    pairs = []
    for root, dirs, files in os.walk(dataset_root):
        if not files:
            continue
        rel_parts = os.path.relpath(root, dataset_root).split(os.sep)
        if not any(p in layout["img_dirs"] for p in rel_parts):
            continue
        lbl_dir = _matching_label_dir(dataset_root, root, layout)
        if lbl_dir is None or os.path.abspath(lbl_dir) == os.path.abspath(root):
            continue
        labels_by_stem = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(lbl_dir, "*"))
            if os.path.isfile(p)
        }
        for img_path in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isfile(img_path):
                continue
            stem = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = labels_by_stem.get(stem)
            if lbl_path:
                pairs.append((img_path, lbl_path))
    return pairs


def _read_split_lists(dataset_root):
    """OpenEarthMap (and some LoveDA mirrors) ship train.txt / val.txt filename lists."""
    lists = {}
    for split in ("train", "val"):
        for cand in glob.glob(os.path.join(dataset_root, "**", f"{split}.txt"), recursive=True):
            with open(cand) as f:
                names = {os.path.splitext(os.path.basename(l.strip()))[0] for l in f if l.strip()}
            if names:
                lists.setdefault(split, set()).update(names)
    return lists


def split_pairs(pairs, dataset_root, layout, val_fraction=0.1):
    """Prefer the dataset's own split (txt lists, then directory markers); fall back to a
    deterministic random split so an unfamiliar layout still trains."""
    lists = _read_split_lists(dataset_root)
    if "train" in lists and "val" in lists:
        train = [p for p in pairs if os.path.splitext(os.path.basename(p[0]))[0] in lists["train"]]
        val = [p for p in pairs if os.path.splitext(os.path.basename(p[0]))[0] in lists["val"]]
        if train and val:
            return train, val, "split lists (train.txt/val.txt)"

    def marked(path, markers):
        parts = {p.lower() for p in path.replace("\\", "/").split("/")}
        return any(m.lower() in parts for m in markers)

    train = [p for p in pairs if marked(p[0], layout["train_markers"])]
    val = [p for p in pairs if marked(p[0], layout["val_markers"])]
    if train and val:
        return train, val, "directory markers"

    shuffled = sorted(pairs)
    random.Random(SEED).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val], f"random {int(val_fraction * 100)}% holdout"


def discover_dataset(data_root, dataset):
    layout = DATASET_LAYOUTS[dataset]
    root = find_dataset_root(data_root, layout["root_names"])
    if root is None:
        return None
    pairs = collect_pairs(root, layout)
    if not pairs:
        return {"root": root, "train": [], "val": [], "how": "no image/label pairs found"}
    train, val, how = split_pairs(pairs, root, layout)
    return {"root": root, "train": train, "val": val, "how": how}


# ==========================================
# 4. AUGMENTATION
#    Image + mask stay in lock-step; the mask is only ever touched with NEAREST
#    resampling so class ids are never interpolated into nonexistent classes.
# ==========================================
class JointTrainTransform:
    """Random scale -> random crop -> flips -> 90-degree rotations -> color jitter.

    Random cropping (rather than squashing a 4096x2160 UAVid frame down to 512x512)
    is what keeps thin, safety-critical classes — Person, Pole — from being resampled
    out of existence before the model ever sees them.
    """

    def __init__(self, crop_size=CROP_SIZE, scale_range=(0.75, 1.5)):
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

    def __call__(self, image: Image.Image, label: np.ndarray):
        mask = Image.fromarray(label, mode="L")

        scale = random.uniform(*self.scale_range)
        # Never scale below the crop size, or padding would dominate the sample.
        min_scale = self.crop_size / min(image.size)
        scale = max(scale, min_scale)
        new_size = (max(1, int(round(image.height * scale))), max(1, int(round(image.width * scale))))
        image = TF.resize(image, new_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, new_size, interpolation=TF.InterpolationMode.NEAREST)

        top = random.randint(0, max(0, image.height - self.crop_size))
        left = random.randint(0, max(0, image.width - self.crop_size))
        image = TF.crop(image, top, left, self.crop_size, self.crop_size)
        mask = TF.crop(mask, top, left, self.crop_size, self.crop_size)

        if random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() < 0.5:  # valid for nadir / near-nadir UAV imagery
            image, mask = TF.vflip(image), TF.vflip(mask)

        k = random.choice([0, 1, 2, 3])  # discrete 90-degree steps only -> no resampling artifacts
        if k:
            image = TF.rotate(image, 90 * k)
            mask = TF.rotate(mask, 90 * k, fill=IGNORE_INDEX)

        image = self.color_jitter(image)  # color-only, never touches the mask
        return image, np.array(mask, dtype=np.uint8)


class JointValTransform:
    """Deterministic resize to a fixed size so validation batches collate."""

    def __init__(self, size=VAL_SIZE):
        self.size = (size, size)

    def __call__(self, image: Image.Image, label: np.ndarray):
        mask = Image.fromarray(label, mode="L")
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, np.array(mask, dtype=np.uint8)


class UnifiedDataset(Dataset):
    """One dataset class for every source. Each sample carries its dataset identity so the
    correct remapping function runs before the mask ever reaches the model, guaranteeing a
    single label space regardless of origin."""

    def __init__(self, pairs, processor, dataset: str, augment: bool):
        if dataset not in REMAP_FNS:
            raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {list(REMAP_FNS)}.")
        self.pairs = pairs
        self.processor = processor
        self.dataset = dataset
        self.remap = REMAP_FNS[dataset]
        self.transform = JointTrainTransform() if augment else JointValTransform()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.remap(_load_label_array(label_path))

        if label.shape[:2] != (image.height, image.width):
            image = image.resize((label.shape[1], label.shape[0]), Image.BILINEAR)

        image, label = self.transform(image, label)

        # do_resize / do_reduce_labels are both off: geometry is fully handled above, and
        # reduce_labels would shift every class id down by one and silently destroy the masks.
        encoded = self.processor(images=image, segmentation_maps=label, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0).long(),
        }


def build_processor():
    return SegformerImageProcessor.from_pretrained(
        PRETRAINED_CKPT, do_resize=False, do_reduce_labels=False
    )


# ==========================================
# 5. CLASS WEIGHTS & METRICS
# ==========================================
def compute_class_weights(dataset_pairs, max_samples=CLASS_WEIGHT_SAMPLES):
    """Log-smoothed inverse-frequency weights from pixel counts.

    dataset_pairs: list of (dataset_name, pairs). Classes absent from the stage get weight
    1.0 — cross-entropy never reads them, and a huge weight there would be meaningless.
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for dataset, pairs in dataset_pairs:
        remap = REMAP_FNS[dataset]
        sample = pairs if len(pairs) <= max_samples else random.sample(pairs, max_samples)
        for _, label_path in tqdm(sample, desc=f"  class stats [{dataset}]", leave=False):
            label = remap(_load_label_array(label_path))
            valid = label[label != IGNORE_INDEX]
            counts += np.bincount(valid.ravel(), minlength=NUM_CLASSES)

    present = counts > 0
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    if present.any():
        freq = counts / max(counts.sum(), 1)
        w = 1.0 / np.log(1.02 + freq)      # log-smoothed inverse frequency, avoids extreme ratios
        w = w / w[present].mean()          # normalize around 1.0 over the present classes
        weights[present] = np.clip(w[present], 0.2, 10.0)
    for c in UNUSED_CLASSES:
        weights[c] = 0.0                   # class 0 is the Ignore slot; nothing should target it
    return torch.tensor(weights, dtype=torch.float32), counts


def update_confusion_matrix(cm, preds: torch.Tensor, labels: torch.Tensor):
    preds = preds.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    valid = labels != IGNORE_INDEX
    preds, labels = preds[valid], labels[valid]
    idx = NUM_CLASSES * labels.astype(np.int64) + preds.astype(np.int64)
    cm += np.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return cm


def iou_per_class(cm):
    intersection = np.diag(cm)
    union = cm.sum(1) + cm.sum(0) - intersection
    ious = np.where(union > 0, intersection / np.maximum(union, 1), np.nan)
    return ious


def traversability_iou(cm):
    """Binary traversable / non-traversable IoU, obtained by collapsing the 14x14 confusion
    matrix into 2x2. This is the metric closest to what the cost map and global planner
    actually consume: a Tree predicted as a Building costs nothing here, but a Rubble pile
    predicted as Road is exactly the error that drives a rover into an obstacle."""
    T, N = TRAVERSABLE_IDS, NON_TRAVERSABLE_IDS
    tt = cm[np.ix_(T, T)].sum()
    tn = cm[np.ix_(T, N)].sum()
    nt = cm[np.ix_(N, T)].sum()
    nn = cm[np.ix_(N, N)].sum()
    ious = []
    if tt + tn + nt > 0:
        ious.append(tt / (tt + tn + nt))
    if nn + tn + nt > 0:
        ious.append(nn / (nn + tn + nt))
    return float(np.mean(ious)) if ious else float("nan")


def summarize_ious(cm):
    """mIoU over classes that actually occur in this stage's val set.

    Averaging over all 14 slots would punish a stage for classes its dataset simply does
    not contain (LoveDA has no Pavement), making checkpoints incomparable across stages.
    """
    ious = iou_per_class(cm)
    support = cm.sum(1)
    scored = np.array([
        i for i in range(NUM_CLASSES)
        if i not in UNUSED_CLASSES and support[i] > 0 and not np.isnan(ious[i])
    ], dtype=np.int64)
    mean_iou = float(np.mean(ious[scored])) if scored.size else 0.0

    crit = [CLASS_NAMES.index(c) for c in CRITICAL_CLASSES if CLASS_NAMES.index(c) in scored]
    critical_iou = float(np.mean(ious[crit])) if crit else float("nan")
    trav_iou = traversability_iou(cm)

    # Checkpoint selection is deliberately weighted toward the drivable-surface classes and
    # the binary traversability split rather than raw mIoU: the best checkpoint for planning
    # is not necessarily the best general scene parser.
    selection = 0.3 * mean_iou
    selection += 0.5 * critical_iou if crit else 0.5 * mean_iou
    selection += 0.2 * trav_iou if not np.isnan(trav_iou) else 0.2 * mean_iou
    return ious, mean_iou, critical_iou, trav_iou, selection, scored, crit


# ==========================================
# 6. STAGE TRAINING
# ==========================================
def run_stage(stage, accelerator, data_root, init_from):
    """Train one progressive stage. Returns the directory holding its best checkpoint,
    which becomes the initialization for the next stage."""
    name = stage["name"]
    stage_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(stage_dir, exist_ok=True)

    accelerator.print("\n" + "=" * 70)
    accelerator.print(f"STAGE {name}  |  datasets: {', '.join(stage['datasets'])}  |  init: {init_from}")
    accelerator.print("=" * 70)

    processor = build_processor()

    train_sets, val_sets, weight_inputs = [], [], []
    for dataset in stage["datasets"]:
        info = discover_dataset(data_root, dataset)
        if info is None or not info["train"]:
            accelerator.print(f"  !! {dataset}: nothing found under {data_root} — skipping")
            continue
        accelerator.print(
            f"  {dataset}: {len(info['train'])} train / {len(info['val'])} val "
            f"({info['how']}) @ {info['root']}"
        )
        train_sets.append(UnifiedDataset(info["train"], processor, dataset, augment=True))
        if info["val"]:
            val_sets.append(UnifiedDataset(info["val"], processor, dataset, augment=False))
        weight_inputs.append((dataset, info["train"]))

    if not train_sets:
        accelerator.print(f"  Stage {name} has no data — skipping.")
        return init_from
    if not val_sets:
        raise RuntimeError(f"Stage {name} found training data but no validation data.")

    train_ds = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_ds = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=NUM_WORKERS > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )

    accelerator.print("  Computing class weights from a training sample...")
    class_weights, counts = compute_class_weights(weight_inputs)
    accelerator.print("  Pixel share: " + ", ".join(
        f"{n}={c / max(counts.sum(), 1) * 100:.2f}%" for n, c in zip(CLASS_NAMES, counts) if c > 0
    ))
    accelerator.print("  Class weights: " + ", ".join(
        f"{n}={w:.2f}" for n, w in zip(CLASS_NAMES, class_weights.tolist())
    ))

    id2label = {i: n for i, n in enumerate(CLASS_NAMES)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        init_from,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id={n: i for i, n in id2label.items()},
        ignore_mismatched_sizes=True,
    )

    # Differential LR: the pretrained encoder gets a smaller LR than the decode head,
    # which stabilizes fine-tuning and limits catastrophic forgetting across stages.
    backbone_params = [p for n, p in model.named_parameters() if "decode_head" not in n]
    head_params = [p for n, p in model.named_parameters() if "decode_head" in n]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": LR_BACKBONE},
         {"params": head_params, "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    class_weights = class_weights.to(accelerator.device)

    # Scheduler steps computed AFTER prepare() so they reflect the real per-process step
    # count under distributed sampling.
    epochs = stage["epochs"]
    total_steps = len(train_loader) * epochs
    lr_scheduler = accelerator.prepare(get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    ))

    log_csv_path = os.path.join(stage_dir, "training_log.csv")
    if accelerator.is_main_process:
        with open(log_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss", "mean_iou", "surface_iou", "traversability_iou"]
                + [f"iou_{c}" for c in CLASS_NAMES]
            )

    best_metric = -float("inf")
    epochs_without_improvement = 0
    patience = stage["patience"]

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc=f"[{name}] Epoch {epoch + 1:03d}/{epochs:03d} [Train]",
            disable=not accelerator.is_main_process, leave=False,
        )
        for step, batch in pbar:
            with accelerator.accumulate(model):
                logits = model(pixel_values=batch["pixel_values"]).logits  # (B, C, h, w), lower res than labels
                labels = batch["labels"]
                upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(upsampled, labels, weight=class_weights, ignore_index=IGNORE_INDEX)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            gathered = accelerator.gather(loss.detach()).mean().item()
            total_train_loss += gathered
            pbar.set_postfix({"iter": f"{gathered:.4f}", "avg": f"{total_train_loss / (step + 1):.4f}"})

        # ---------------- Validation ----------------
        model.eval()
        total_val_loss = 0.0
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        vbar = tqdm(
            val_loader, desc=f"[{name}] Epoch {epoch + 1:03d}/{epochs:03d} [Val]  ",
            disable=not accelerator.is_main_process, leave=False,
        )
        with torch.no_grad():
            for batch in vbar:
                logits = model(pixel_values=batch["pixel_values"]).logits
                labels = batch["labels"]
                upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(upsampled, labels, weight=class_weights, ignore_index=IGNORE_INDEX)
                total_val_loss += accelerator.gather(loss.detach()).mean().item()

                preds = accelerator.gather_for_metrics(upsampled.argmax(dim=1))
                gt = accelerator.gather_for_metrics(labels)
                if accelerator.is_main_process:
                    cm = update_confusion_matrix(cm, preds, gt)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        if accelerator.is_main_process:
            ious, mean_iou, critical_iou, trav_iou, selection, scored, crit = summarize_ious(cm)
            crit_names = "/".join(CLASS_NAMES[i] for i in crit) or "none present"
            accelerator.print(
                f"[{name}] Epoch [{epoch + 1:03d}/{epochs:03d}] -> "
                f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | "
                f"mIoU {mean_iou:.4f} | Surface ({crit_names}) {critical_iou:.4f} | "
                f"Traversability {trav_iou:.4f}"
            )
            for i in scored:
                tag = "drivable" if i in TRAVERSABLE_IDS else "obstacle"
                accelerator.print(f"    {CLASS_NAMES[i]:12s} IoU: {ious[i]:.4f}  ({tag})")

            with open(log_csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch + 1, avg_train_loss, avg_val_loss, mean_iou, critical_iou, trav_iou]
                    + [0.0 if np.isnan(v) else float(v) for v in ious]
                )

            if selection > best_metric:
                best_metric = selection
                epochs_without_improvement = 0
                accelerator.unwrap_model(model).save_pretrained(stage_dir)
                processor.save_pretrained(stage_dir)
                with open(os.path.join(stage_dir, "stage_info.json"), "w") as f:
                    json.dump({
                        "stage": name, "datasets": stage["datasets"], "init_from": init_from,
                        "epoch": epoch + 1, "mean_iou": mean_iou, "surface_iou": critical_iou,
                        "traversability_iou": trav_iou, "selection_metric": best_metric,
                        "class_names": CLASS_NAMES,
                    }, f, indent=2)
                accelerator.print(f"  New best checkpoint -> {stage_dir} (selection={best_metric:.4f})")
            else:
                epochs_without_improvement += 1
                accelerator.print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")
            accelerator.print("-" * 60)

        # Broadcast the stop decision so every process stays in sync under multi-GPU.
        stop = torch.tensor(
            [1 if (accelerator.is_main_process and epochs_without_improvement >= patience) else 0],
            device=accelerator.device,
        )
        if accelerator.reduce(stop, reduction="sum").item() > 0:
            accelerator.print(f"[{name}] Early stopping at epoch {epoch + 1}.")
            break

    accelerator.print(f"[{name}] done. Best selection metric: {best_metric:.4f}")
    return stage_dir


# ==========================================
# 7. DATA CHECK (run this once the archives finish extracting)
# ==========================================
def check_data(data_root):
    print(f"Scanning {os.path.abspath(data_root)}\n")
    for dataset in DATASET_LAYOUTS:
        info = discover_dataset(data_root, dataset)
        print(f"[{dataset}]")
        if info is None:
            print("  root not found\n")
            continue
        print(f"  root : {info['root']}")
        print(f"  pairs: {len(info['train'])} train / {len(info['val'])} val  ({info['how']})")
        if not info["train"]:
            print()
            continue
        img, lbl = info["train"][0]
        print(f"  sample image: {img}")
        print(f"  sample label: {lbl}")
        try:
            raw = _load_label_array(lbl)
            print(f"  raw mask shape={raw.shape} dtype={raw.dtype} "
                  f"mode={Image.open(lbl).mode} unique={np.unique(raw)[:12]}")
            counts = np.zeros(NUM_CLASSES + 1, dtype=np.int64)
            sample = info["train"][:30]
            for _, p in sample:
                m = REMAP_FNS[dataset](_load_label_array(p))
                # bucket IGNORE_INDEX into slot NUM_CLASSES so it shows up in the report
                flat = np.where(m == IGNORE_INDEX, NUM_CLASSES, m).ravel()
                counts += np.bincount(flat, minlength=NUM_CLASSES + 1)
            total = max(counts.sum(), 1)
            print("  remapped class share over 30 masks:")
            for i, c in enumerate(counts):
                if c == 0:
                    continue
                nm = "IGNORED" if i == NUM_CLASSES else f"{i} {CLASS_NAMES[i]}"
                print(f"    {nm:16s} {c / total * 100:6.2f}%")
        except Exception as e:
            print(f"  !! failed to read/remap: {e}")
        print()


# ==========================================
# 8. ENTRY POINT
# ==========================================
STAGES = [
    {"name": "stage1_openearthmap", "datasets": ["openearthmap"], "epochs": 60, "patience": 10},
    {"name": "stage2_loveda",       "datasets": ["loveda"],       "epochs": 60, "patience": 10},
    {"name": "stage3_uavid",        "datasets": ["uavid"],        "epochs": 80, "patience": 12},
    {"name": "stage4_field",        "datasets": ["field"],        "epochs": 100, "patience": 15},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--stages", default=None,
                        help="Comma-separated subset, e.g. 'openearthmap,uavid'. Default: all.")
    parser.add_argument("--init-from", default=None,
                        help="Checkpoint to start the first stage from. Default: %s" % PRETRAINED_CKPT)
    parser.add_argument("--check-data", action="store_true",
                        help="Verify dataset discovery and label remapping, then exit.")
    args = parser.parse_args()

    if args.check_data:
        check_data(args.data_dir)
        return

    stages = STAGES
    if args.stages:
        wanted = [s.strip() for s in args.stages.split(",")]
        stages = [s for s in STAGES if any(w in s["name"] or w in s["datasets"] for w in wanted)]
        if not stages:
            raise SystemExit(f"No stages matched {wanted}. Available: {[s['name'] for s in STAGES]}")

    set_seed(SEED)
    accelerator = Accelerator(mixed_precision="bf16")  # switch to "fp16" on pre-Ampere GPUs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    init_from = args.init_from or PRETRAINED_CKPT
    for stage in stages:
        init_from = run_stage(stage, accelerator, args.data_dir, init_from)

    accelerator.print(f"\nAll stages complete. Final checkpoint: {init_from}")


if __name__ == "__main__":
    main()

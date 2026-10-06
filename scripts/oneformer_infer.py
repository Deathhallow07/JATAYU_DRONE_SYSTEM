import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    OneFormerProcessor,
    OneFormerForUniversalSegmentation,
)

# ==========================================================
# CONFIG
# ==========================================================
IMAGE_PATH = "/media/uasdtu/DataSets2/Segmentation_GATE_1/frame_selector_bag_output/frame_1772267357_497718262_seq6229.jpg"

# HuggingFace checkpoint
MODEL_NAME = "shi-labs/oneformer_ade20k_swin_large"

# # ADE20K (best general scene understanding)
# MODEL_NAME = "shi-labs/oneformer_ade20k_swin_large"

# # COCO (more object-centric)
# MODEL_NAME = "shi-labs/oneformer_coco_swin_large"

# # Cityscapes (roads/urban scenes)
# MODEL_NAME = "shi-labs/oneformer_cityscapes_swin_large"

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"



NAV_CLASSES = {
    "PAVEMENT": [
        "road",
        "sidewalk",
        "path",
        "runway",
        "floor",
        "bridge",
        "stairs",
        "stairway",
    ],

    "GRASS": [
        "grass",
        "field",
    ],

    "SOIL": [
        "earth",
        "earth, ground",
        "sand",
    ],

    "TREE": [
        "tree",
        "palm",
    ],

    "VEGETATION": [
        "plant",
        "flower",
    ],

    "BUILDING": [
        "building",
        "house",
        "skyscraper",
    ],

    "WATER": [
        "water",
        "sea",
        "river",
    ],

    "OBSTACLE": [
        "wall",
        "fence",
        "rock",
        "column",
        "railing",
        "signboard",
        "pole",
        "barrier",
    ],

    "VEHICLE": [
        "car",
        "bus",
        "truck",
        "boat",
        "airplane",
    ],

    "PERSON": [
        "person",
    ],
}

NAV_IDS = {
    "UNKNOWN": 0,
    "PAVEMENT": 1,
    "GRASS": 2,
    "SOIL": 3,
    "TREE": 4,
    "VEGETATION": 5,
    "BUILDING": 6,
    "WATER": 7,
    "OBSTACLE": 8,
    "VEHICLE": 9,
    "PERSON": 10,
}

NAV_NAMES = {v: k for k, v in NAV_IDS.items()}

# ==========================================================
# LOAD MODEL
# ==========================================================
print("Loading OneFormer...")

processor = OneFormerProcessor.from_pretrained(MODEL_NAME)
model = OneFormerForUniversalSegmentation.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("Model loaded.")


id_to_nav = {}

for idx, name in model.config.id2label.items():

    assigned = "UNKNOWN"

    for nav_class, ade_classes in NAV_CLASSES.items():

        if name in ade_classes:
            assigned = nav_class
            break

    id_to_nav[idx] = assigned

print("\nADE20K -> Navigation Mapping\n")

for nav in NAV_CLASSES:

    print(f"\n{nav}")

    for idx, name in model.config.id2label.items():

        if id_to_nav[idx] == nav:
            print(f"   {idx:3d} : {name}")
    
# ==========================================================
# LOAD IMAGE
# ==========================================================
bgr = cv2.imread(IMAGE_PATH)

if bgr is None:
    raise FileNotFoundError(IMAGE_PATH)

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
pil = Image.fromarray(rgb)

H, W = rgb.shape[:2]

# ==========================================================
# INFERENCE
# ==========================================================
with torch.no_grad():

    inputs = processor(
        images=pil,
        task_inputs=["semantic"],
        return_tensors="pt",
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs)

prediction = processor.post_process_semantic_segmentation(
    outputs,
    target_sizes=[(H, W)],
)[0]

prediction = prediction.cpu().numpy()

print("\n==============================")
print("ADE20K CLASSES PRESENT")
print("==============================")

for cls in np.unique(prediction):
    print(f"{cls:3d} : {model.config.id2label[int(cls)]}")

nav_prediction = np.zeros_like(prediction, dtype=np.uint8)

for ade_id, nav_name in id_to_nav.items():
    nav_prediction[prediction == ade_id] = NAV_IDS[nav_name]

print("\n==============================")
print("UNKNOWN ADE20K CLASSES")
print("==============================")

unknown_ids = np.unique(prediction[nav_prediction == NAV_IDS["UNKNOWN"]])

for idx in unknown_ids:
    print(f"{idx:3d} : {model.config.id2label[int(idx)]}")


# ==========================================================
# COLORIZE
# ==========================================================

NAV_PALETTE = np.array([
    [0,0,0],          # UNKNOWN
    [160,160,160],    # PAVEMENT
    [0,220,0],        # GRASS
    [139,69,19],      # SOIL
    [34,139,34],      # TREE
    [0,180,0],        # VEGETATION
    [255,0,0],        # BUILDING
    [0,0,255],        # WATER
    [255,255,0],      # OBSTACLE
    [255,128,0],      # VEHICLE
    [255,0,255],      # PERSON
], dtype=np.uint8)

mask = NAV_PALETTE[nav_prediction]


# ==========================================================
# OVERLAY
# ==========================================================
overlay = cv2.addWeighted(
    bgr,
    0.45,
    mask[..., ::-1],
    0.55,
    0,
)


# ==========================================================
# PRINT DETECTED CLASSES + THEIR COLORS
# ==========================================================
classes = np.unique(nav_prediction)

print("\nDetected Navigation Classes\n")

total_pixels = nav_prediction.size

print(f"{'ID':>3} {'Class':<18} {'Pixels':>12} {'Percent':>10} {'Color'}")
print("-" * 75)

for c in classes:

    pixels = np.sum(nav_prediction == c)
    percent = 100.0 * pixels / total_pixels

    color = NAV_PALETTE[c]

    print(
        f"{c:3d} "
        f"{NAV_NAMES[c]:<18}"
        f"{pixels:12d}"
        f"{percent:9.2f}%   "
        f"RGB({color[0]:3d},{color[1]:3d},{color[2]:3d})"
    )

# ==========================================================
# CREATE LEGEND
# ==========================================================
legend = np.ones((35 * len(classes), 350, 3), dtype=np.uint8) * 255

for i, cls in enumerate(classes):

    color = tuple(int(x) for x in NAV_PALETTE[cls][::-1])   # RGB -> BGR

    y = i * 35

    cv2.rectangle(
        legend,
        (10, y + 5),
        (40, y + 30),
        color,
        -1,
    )

    cv2.putText(
        legend,
        f"{cls}: {NAV_NAMES[int(cls)]}",
        (55, y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        ade = int(prediction[y, x])
        nav = int(nav_prediction[y, x])

        print("\n-----------------------------")
        print(f"Pixel : ({x}, {y})")
        print(f"ADE20K : {ade} -> {model.config.id2label[ade]}")
        print(f"NAV    : {NAV_NAMES[nav]}")
        print("-----------------------------")

cv2.namedWindow("Overlay")
cv2.setMouseCallback("Overlay", mouse_callback)

cv2.imshow("Legend", legend)

# ==========================================================
# DISPLAY
# ==========================================================
cv2.imshow("Original", bgr)
cv2.imshow("Segmentation Mask", mask[..., ::-1])
cv2.imshow("Overlay", overlay)

print("\nPress any key to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
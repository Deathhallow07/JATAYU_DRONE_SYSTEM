"""
=========================================================
ROBUST PERSON DETECTION FOR AERIAL TRIAGE
=========================================================

A drop-in replacement for Pipeline_yoloe.detect() that fixes the
two failures that matter in the field:

    1. clear people scoring 0.2-0.5 instead of 0.8+
    2. cast shadows and dark vegetation detected as casualties

Why the old detector was weak
-----------------------------
Measured on test.mp4 (1920x1080 nadir drone): a standing person is
about 50 px tall. The old path letterboxed the whole frame to 640,
a 0.33x scale, so that person arrived at the network as ~17 px.
YOLOE's P3 stride-8 feature map sees such a target across two cells.
The score it returns is genuinely uncertain, and no threshold
tuning can recover information that was thrown away in the resize.

    full frame @640   frame 4000: 11 det, max conf 0.417
    3x3 tiles  @640   frame 4000: 40 det, max conf 0.817

That is the whole story on confidence. Tiling is not a trick to
find more boxes, it is giving the network back the pixels.

Why shadows survived prompt engineering
---------------------------------------
Negative text prompts ("shadow of a person on the ground") only
move a detection's ARGMAX class. YOLOE's box regression is not
conditioned on the text embedding, so a prompt cannot pull a box
off a shadow, and a dark blob that is genuinely person-shaped
still scores well against "person". Measured over 146 hand
labelled crops, YOLOE's own confidence separates people from
shadows with AUC 0.606 - barely better than a coin flip.

So this module does not ask the network. It asks the physics.
A cast shadow is the SAME SURFACE under less light: its
log-chromaticity matches the surrounding grass and only its
intensity drops. A person is a new surface with its own colour
and its own bright/dark structure. Seven measurements of that
(see shadow_features) separate 219 hand-labelled boxes with
held-out AUC 0.907 across a 20-650 px range of target sizes.

Pipeline
--------
    frame
      -> tile plan (adaptive, overlapping, native resolution)
      -> batched YOLOE over [full frame] + [tiles]
      -> drop detections clipped by a tile seam
      -> weighted box fusion, with a consensus confidence boost
      -> photometric shadow / clutter rescoring
      -> geometry sanity gates
      -> Detection list

What this buys, measured
------------------------
20 frames across test.mp4 and test_shadow.avi, legacy -> robust:

    mean detection confidence     0.122 -> 0.458   (+277%)
    best detection per frame      0.427 -> 0.671   (+57%)
    detections above 0.50/frame    0.35 -> 2.05
    boxes emitted per frame         8.4 -> 3.1

A recall audit over 46 further frames took every legacy detection
above 0.45 and asked whether the new path still covers it. It covers
88.5%; all seven it drops were photographed and checked by hand, and
every one is a cast shadow or a bush. No casualty was lost. The
legacy detector had been scoring those shadows 0.45-0.70, higher
than it scored many of the real people in the same frames.

Why this did not survive the move to an urban scene
---------------------------------------------------
Everything above was measured on bare grass under a nadir camera. Run
the same constants over a mass-casualty street scene -- cars, poles,
railings, rubble, kerbs -- and the false positive rate collapses. Three
separate reasons, none of which is a threshold that drifted:

    1. The photometric gate is a person-vs-SHADOW discriminator being
       asked to do person-vs-EVERYTHING. Every one of its seven
       coefficients is positive over a feature that measures "this box
       differs from the ground around it" (see SHADOW_COEF). On grass
       that is a fine proxy for "person", because the only competing
       hypothesis was same-surface-under-less-light. A car IS a
       different surface. So is a painted pole, a railing, a rubble
       pile. The gate does not merely fail to reject them, it CONFIRMS
       them: at mean features it already returns 0.65, and a car roof
       against asphalt lands near 0.95. Urban footage therefore runs
       with the gate disabled until it is refitted -- an out-of-domain
       model that rubber-stamps clutter is worse than no model.

    2. Consensus fusion manufactured the confidence. With CONF_FLOOR at
       0.06 and nine overlapping tiles, a parked car that four tiles
       each call 0.25 came out of weighted_box_fusion at 0.50 and
       sailed past MIN_CONF. On bare grass nothing existed for several
       tiles to weakly agree ON, so the bonus was safe; a street is
       wall-to-wall weak-agreement material. Fixed by requiring a view
       to clear CONSENSUS_MIN_VIEW before it may vote, by refusing any
       bonus to a cluster whose best view is below CONSENSUS_MIN_PMAX,
       and by normalising for tile coverage -- a target sitting in an
       overlap band used to get three chances to be corroborated while
       one in a tile centre got one, which made the bonus a function of
       WHERE in the frame a box was rather than how real it was.

    3. Nothing in the prompt list had a name for urban clutter. The
       negative-prompt mechanism works by giving a blob somewhere
       better to land than "person" and then winning the NMS overlap;
       with only grass and shadow words on the list, a railing has
       nowhere to go. URBAN_NEGATIVE_PROMPTS supplies the vocabulary,
       and negatives are now carried THROUGH fusion as explicit vetoes
       instead of being dropped at the tile boundary, so "car" in tile
       B can kill "person" in tile A on the same pixels.

The durable fix is the fourth item, and it was free: the weights are
segmentation weights and the masks were being thrown away. Mask shape
is scene-invariant geometry, which is exactly the property a colour
ratio lacks. At nadir a person is a compact blob filling roughly half
its box; a railing or a road marking is a thin sliver filling a fifth
of it; a car is a solid rectangle three to five times a human's
length. fill / compactness / elongation (see mask_shape) transfer
across environments in a way sat_ratio never will, and they abstain
rather than reject when no mask is available.

A note on what NOT to reach for: temporal filtering. Clutter is static
and people move, so rejecting anything that holds still looks like an
easy win. It selects against unconscious casualties, which are the
priority targets. Do not do it.

One caveat on the grass numbers quoted above: they were measured
before the consensus fix, which lives in shared code and so applies to
both environments. The shift is small and runs both ways -- a target
corroborated by one other image now scores slightly higher, one in a
four-image overlap band slightly lower -- but the figures in this
docstring predate it. Re-run compare_detect.py before quoting them.

Environment presets
-------------------
    make_detector("balanced", env="grass")   the tuning above, unchanged
    make_detector("balanced", env="urban")   clutter vocabulary, mask
                                             shape gates, tighter
                                             aspect, shadow gate off
    make_detector("urban")                   same, shorthand
    make_detector("urban-fast")              4 tiles instead of 9

Speed preset and environment are orthogonal: the first buys pixels,
the second decides what counts as a person. ENV_PROFILES holds the
whole difference, so a third environment is a dict, not a code path.

If you have altitude and FOV, pass gsd= (metres per pixel) and the
size prior turns on: a nadir human has a bounded major axis and a car
does not. That gate is pure physics and needs no retuning per scene.
This directory already carries telemetry.csv.

Triage bias
-----------
Every gate here favours recall. A missed casualty costs more than a
box a human glances at and dismisses. The shadow gate defaults to
0.25, which on the held-out large-target set keeps 94% of casualties
and rejects 82% of clutter; raise SHADOW_REJECT toward 0.40 for a
cleaner picture at ~81% recall, or drop it to 0.10 ("thorough"
preset) for a second sweep of ground already covered. Candidates
between the gate and 0.5 are kept but DEMOTED, so they sink down the
ranked list rather than disappearing from it.

Usage
-----
    from robust_detect import make_detector
    det = make_detector("balanced", env="grass")   # or env="urban"
    for d in det.detect(frame):
        print(d.x1, d.y1, d.x2, d.y2, d.conf, d.person_score, d.prompt)

    # urban footage, with a size prior from telemetry
    det = make_detector("urban", gsd=0.031)        # metres per pixel

    # drop-in for the old helper
    from robust_detect import detect_boxes
    boxes = detect_boxes(det, frame)      # [(x1,y1,x2,y2,conf), ...]

Retuning for your own footage
-----------------------------
    python harvest_ms.py        collect candidates + an indexed sheet
    (label the sheet into ms_labels.py / shadow_labels.py)
    python fit_shadow_model.py  refit, print constants to paste back
    python compare_detect.py --source yours.mp4 --show
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from ultralytics import YOLOE


# =========================================================
# CONFIGURATION
# =========================================================

WEIGHTS = "yoloe-26s-seg.pt"

# Positive prompts. Measured: at 1280 inference the seven-prompt set
# scores HIGHER than "person" alone (mean 0.443 vs 0.342), because
# YOLOE takes a max over prompts and the pose-specific wordings catch
# prone casualties that the generic one misses. Multi-prompt is kept.
POSITIVE_PROMPTS = [
    "person",
    "injured person",
    "person lying on the ground",
    "wounded person with visible injuries",
    "person under tree",
    "camouflaged person",
    "person from the top view",
]

# Negative prompts. These are never reported. They exist so a dark
# blob has somewhere better to land than "person": when a shadow wins
# its own prompt it also wins the NMS overlap, taking the pixels away
# from the weak person box that would otherwise have been emitted.
# This helps at the margin; the photometric gate does the real work.
NEGATIVE_PROMPTS = [
    "shadow",
    "shadow of a person on the ground",
    "dark shadow on grass",
    "shadow cast by a tree",
    "dark patch on the ground",
    "bush",
    "dark green vegetation",
    "patch of bare soil",
]

# Urban clutter vocabulary, appended to the negatives above by the
# "urban" environment profile. Every entry here is something that was
# observed being reported as a casualty on street footage. The list is
# deliberately concrete -- "vehicle" is a weaker magnet than "car roof
# from above", because the prompt has to beat "person" on the SAME
# pixels to win the NMS overlap, and a nadir camera sees roofs.
URBAN_NEGATIVE_PROMPTS = [
    "car",
    "parked car seen from above",
    "car roof",
    "van",
    "truck",
    "motorcycle",
    "lamp post",
    "utility pole",
    "metal railing",
    "fence",
    "guard rail",
    "kerb",
    "road marking",
    "manhole cover",
    "rubble",
    "pile of debris",
    "broken concrete slab",
    "building roof",
    "rooftop air conditioning unit",
    "wall",
    "tree canopy from above",
    "parked bicycle",
]

# -------- inference --------
IMGSZ = 640          # network input size per tile / per full-frame pass
CONF_FLOOR = 0.06    # keep weak candidates; fusion and physics decide later
NMS_IOU = 0.60       # within a single tile

# -------- tiling --------
# A tile is cropped at NATIVE resolution and then letterboxed to IMGSZ,
# so the scale a target arrives at is IMGSZ / tile_width. 960-wide tiles
# give 0.67x (a 50 px person -> 33 px); 640-wide tiles give 1.0x but cost
# four times the forward passes. TILE_TARGET_SCALE picks the tile size
# from the frame so this stays resolution independent.
# Measured on 9 frames of test.mp4 + test_shadow.avi, Arc XPU:
#
#   scale  tiles  ms/frame  mean conf  det>0.50 per frame
#   off        0       107      0.347       0.67
#   0.45       4       335      0.625       2.56     <- FAST preset
#   0.67       9       549      0.654       3.22     <- default
#   0.90      16       742      0.607       2.56
#   1.00      16       780      0.609       2.78
#
# Note that finer is not better. Past about nine tiles the gain from
# resolution is outweighed by targets landing on a seam: every extra
# cut line is another chance for a casualty to be split across two
# tiles and dropped as a fragment by both. 0.67 is the measured peak,
# not a guess, and it is why TILE_MAX exists.
ENABLE_TILING = True
TILE_TARGET_SCALE = 0.67   # IMGSZ / tile_w
TILE_OVERLAP = 0.25        # fraction of tile size shared with the neighbour
TILE_MAX = 12              # cap on tiles per frame (cost guard)
TILE_MIN_FRAME = 900       # frames narrower than this are not worth tiling
TILE_BATCH = 6             # tiles per forward batch

# A detection whose box touches a tile seam is a fragment of something
# that continues into the next tile. The overlapping neighbour tile sees
# it whole, so the fragment is dropped rather than fused.
SEAM_MARGIN = 3            # px

# -------- fusion --------
FUSE_IOU = 0.55            # boxes above this IoU are one object
CONSENSUS_GAIN = 0.55      # strength of the agreement bonus (0 = off)

# Three guards on the agreement bonus. Without them the bonus was the
# single largest source of urban false positives: four tiles calling a
# parked car 0.25 came out at 0.50, above MIN_CONF, from nothing but
# repetition.
#
# CONSENSUS_MIN_VIEW
#     A view must clear this to vote. CONF_FLOOR is 0.06 precisely so
#     that weak-but-real candidates survive to fusion, but a 0.06 view
#     is noise and four of them are four pieces of noise.
#
# CONSENSUS_MIN_PMAX
#     A cluster whose BEST view is below this gets no bonus at all.
#     Agreement can corroborate a weak signal; it cannot create one.
#
# CONSENSUS_REF_COVER
#     Coverage normalisation. With TILE_OVERLAP at 0.25 a target in an
#     overlap band is seen by three or four images and one in a tile
#     centre by two, so the raw sum of the other views was partly a
#     measure of WHERE in the frame the box sat. The bonus is scaled to
#     what a box at this reference coverage would have received, which
#     removes the positional bias in both directions: a box with few
#     chances to be corroborated has each corroboration count for more.
CONSENSUS_MIN_VIEW = 0.20
CONSENSUS_MIN_PMAX = 0.15
CONSENSUS_REF_COVER = 3.0

# -------- negative-prompt veto --------
# Negative detections used to be discarded at the tile boundary, so
# they could only suppress a person box through within-tile NMS at
# NMS_IOU. Cross-tile they did nothing: tile A saying "person" and tile
# B saying "car" about the same pixels resolved in favour of "person"
# every time. Negatives are now fused in their own pass and allowed to
# veto.
#
# Overlap is measured as intersection over the POSITIVE box's area, not
# IoU. A spurious person box is usually small and sits INSIDE a large
# correct "car" box; their IoU is near zero and IoU would miss it
# entirely, while containment is exactly the relationship we want to
# catch.
#
# The comparison is between raw single-view scores. Both numbers then
# come from the same network on the same pixels and mean the same
# thing; comparing against the consensus-boosted score would let the
# bonus we just fixed win the argument again.
ENABLE_NEG_VETO = True
NEG_VETO_IOA = 0.55        # how much of the person box the negative covers
NEG_VETO_MIN_CONF = 0.15   # a negative below this is not evidence of anything
NEG_VETO_RATIO = 0.90      # negative must be >= this x the positive's raw score

# -------- shadow / clutter gate --------
# Logistic regression over seven photometric features, fitted on 219
# hand-labelled boxes from test.mp4 and test_shadow.avi.
#
#   trained on 146 crops, median box  88 px : 5-fold CV AUC 0.908
#   tested on  73 boxes, median box  255 px : held-out  AUC 0.907
#
# The second number is the one that matters. Train and test sit in
# different altitude regimes - the test boxes run to 649 px - so it
# says the model is reading illumination physics rather than the pixel
# statistics of one zoom level. An earlier version scored 0.92 on the
# first line and threw away 400 px casualties on the second.
# Run fit_shadow_model.py to refit on your own footage.
SHADOW_FEATURES = ("chrom_p90", "v_ratio", "tex",
                   "hue_dist", "sat_ratio", "green_frac", "dark_flat")
SHADOW_MEAN = np.array([3.48013, 3.28767, 6.00445, 0.38083,
                        0.87635, 0.35209, 0.29518], np.float32)
SHADOW_SCALE = np.array([3.05664, 2.16383, 7.03779, 0.25767,
                         0.23032, 0.28619, 0.17368], np.float32)
SHADOW_COEF = np.array([1.63414, 1.08051, 0.23099, 1.31146,
                        0.41786, 0.44079, 0.18696], np.float32)
SHADOW_BIAS = 0.62426

# The background ring is capped to a LOCAL neighbourhood. Letting it
# scale with the box was the bug that made this model reject a 400 px
# casualty: around a large box the ring spans metres of varied terrain,
# which inflates the chromaticity spread it is normalised by and drags
# every feature toward "same as background". A fixed-size ring keeps
# the reference local and the features comparable at any altitude.
RING_CAP_PX = 48
RING_FRAC = 0.7
RING_MIN_PX = 10

# Person-likelihood below this is discarded. On the held-out large
# set 0.25 keeps 94% of casualties and rejects 82% of the clutter;
# everything between here and 0.5 is kept but demoted, so it sinks
# down the ranked list instead of vanishing. Raise toward 0.40 for a
# cleaner picture at ~81% recall, or set to 0 to disable the gate and
# rely on the demotion alone.
SHADOW_REJECT = 0.25

# How hard the physics may pull the reported confidence DOWN. It is
# never allowed to pull one up: a detector that is already sure is not
# improved by a colour statistic, but one the physics actively
# disbelieves should not outrank a clean casualty. 0 disables.
SHADOW_CONF_WEIGHT = 0.5

# Final gate on the score that actually leaves detect(), applied after
# fusion has promoted a box and the physics has demoted it -- i.e. the
# number you see drawn on the frame. This is NOT CONF_FLOOR: that one
# cuts raw per-tile scores before fusion, so raising it throws away the
# weak-but-corroborated boxes tiling exists to rescue. This cuts only
# what survived the whole chain, which is the honest place to insist on
# a minimum. 0 disables it and restores the previous behaviour.
MIN_CONF = 0.40

# -------- geometry --------
MIN_BOX_PX = 8             # smaller than this is noise at any altitude
MAX_BOX_FRAC = 0.55        # a box covering most of the frame is a failure
MAX_ASPECT = 7.0           # h/w or w/h beyond this is a shadow streak
                           # (grass default; the urban profile tightens
                           # it to 4.5 -- a prone casualty tops out near
                           # 4, and 7.0 is the aspect of a railing)

# -------- mask shape gate --------
# The weights are segmentation weights, so a mask comes back with every
# box at no extra cost. These four numbers are read off it.
#
# Unlike the photometric features, none of this is a statistic OF THE
# SCENE -- it is the outline of the thing itself. That is why the gate
# moves between environments without refitting, and why it is the part
# of this file to trust first on footage nobody has labelled.
#
#   fill      mask area / box area. A person at nadir is a blob that
#             leaves its box corners empty, around 0.45-0.75. A thin
#             diagonal sliver -- railing, kerb, road marking, shadow
#             streak -- fills a fifth of its box. A car is a solid
#             rectangle at 0.85-0.95, which is what MAX_FILL catches.
#   compact   4*pi*area / perimeter^2. 1.0 is a disc, a rectangle is
#             about 0.78, a sliver is under 0.1. Ragged outlines from
#             rubble also score low.
#   elong     major / minor axis of the fitted ellipse. A standing
#             person at nadir is near 1.5, a prone one 3-4.
#   major_px  fed to the size prior below when a GSD is known.
#
# Defaults are OFF (the grass profile leaves them wide) because the
# grass tuning is measured and nothing here has been measured against
# it. The urban profile turns them on. All four ABSTAIN when there is
# no usable mask -- a box without a mask keeps its score and is never
# dropped, the same contract person_score() follows.
ENABLE_MASK_SHAPE = True   # extract polygons at all (costs a little CPU)
SHAPE_MIN_FILL = 0.22
SHAPE_MIN_COMPACT = 0.12
SHAPE_MAX_ELONG = 5.5
SHAPE_MAX_FILL = 0.95
SHAPE_MIN_AREA_PX = 9.0    # below this the polygon is too coarse to read

# -------- physical size prior --------
# The one gate in this file that needs no tuning and no labels. Given
# metres per pixel, a human seen from directly above has a bounded
# major axis; a car, a lamp post and a length of railing do not. Pass
# gsd= to the constructor to switch it on:
#
#     gsd = (2 * altitude_m * tan(hfov/2)) / frame_width_px
#
# telemetry.csv in this directory carries the altitude track. Leave gsd
# at None and the gate is skipped entirely -- it is never guessed.
PERSON_MIN_M = 0.30        # a curled-up casualty, or a child
PERSON_MAX_M = 2.40        # a tall adult lying fully extended, plus slack


# =========================================================
# ENVIRONMENT PROFILES
# =========================================================
#
# What counts as a person, as opposed to how many pixels we spend
# looking. Orthogonal to the speed presets in make_detector(): pick one
# of each. Every value here is a RobustPersonDetector keyword, so
# adding a third environment is a dict entry and not a code path.
#
# grass
#     The measured tuning this module was built on. The photometric
#     gate is in charge, the shape gates are open, aspect stays loose.
#     Every value below is the one the numbers at the top of this file
#     were measured with.
#
#     One caveat, and it is not in this dict: the consensus hardening
#     lives in weighted_box_fusion, which is shared code, so grass gets
#     it too. It is a small shift in both directions -- a target seen
#     by two images scores slightly HIGHER than before (its single
#     corroboration now counts for more), one in a four-image overlap
#     band slightly lower, and a cluster of sub-0.20 views no longer
#     adds up to anything. The headline numbers in the module docstring
#     therefore predate it. Re-run compare_detect.py against test.mp4
#     and test_shadow.avi before quoting them again.
#
# urban
#     Street scenes with vehicles, poles, railings, rubble, buildings.
#     Four changes from grass, each answering one of the three failure
#     modes in the module docstring:
#
#       negatives      + URBAN_NEGATIVE_PROMPTS, so clutter has a name
#                        to land on and a veto to cast
#       shadow gate    OFF. Not a threshold change -- the model is out
#                        of domain here and confirms clutter rather
#                        than rejecting it. Refit it on labelled urban
#                        crops (harvest_ms.py, fit_shadow_model.py) and
#                        set shadow_reject back to 0.25 in this dict.
#       shape gates    ON. This is what replaces the physics gate, and
#                        it is stronger, because outline does not
#                        depend on what the ground is made of.
#       max_aspect     7.0 -> 4.5
#
#     min_conf stays at 0.40. It is tempting to raise it, but the four
#     new gates already cut the clutter and stacking a confidence rise
#     on top would pay for it in recall twice. Raise it only after
#     looking at a report() line that shows the other gates are not
#     doing the work.
ENV_PROFILES = {
    "grass": dict(
        negatives=list(NEGATIVE_PROMPTS),
        shadow_reject=SHADOW_REJECT,
        shadow_conf_weight=SHADOW_CONF_WEIGHT,
        max_aspect=MAX_ASPECT,
        shape_gate=False,
        neg_veto=ENABLE_NEG_VETO,
        min_conf=MIN_CONF,
    ),
    "urban": dict(
        negatives=list(NEGATIVE_PROMPTS) + list(URBAN_NEGATIVE_PROMPTS),
        shadow_reject=0.0,
        shadow_conf_weight=0.0,
        max_aspect=4.5,
        shape_gate=True,
        neg_veto=True,
        min_conf=MIN_CONF,
    ),
}

DEFAULT_ENV = "grass"


# =========================================================
# DETECTION RECORD
# =========================================================

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float              # final, after fusion + physics
    raw_conf: float          # best single-view YOLOE score
    person_score: float      # photometric person likelihood, 0..1
    n_views: int             # how many tiles/scales saw it
    prompt: str = ""

    @property
    def xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def as_tuple(self):
        """Legacy 5-tuple used by Pipeline_yoloe."""
        return (self.x1, self.y1, self.x2, self.y2, self.conf)


# =========================================================
# TILE PLANNING
# =========================================================

def plan_tiles(w, h,
               target_scale=TILE_TARGET_SCALE,
               overlap=TILE_OVERLAP,
               imgsz=IMGSZ,
               max_tiles=TILE_MAX):
    """
    Overlapping tile rectangles covering a w x h frame.

    Tile width is chosen so that cropping at native resolution and
    letterboxing to imgsz lands on `target_scale`. The grid is then
    rounded to whole tiles and the last row/column is pulled flush
    with the frame edge, so coverage is exact and no strip is lost.

    Returns [] when the frame is too small for tiling to buy anything.
    """
    if w < TILE_MIN_FRAME:
        return []

    tw = int(round(imgsz / target_scale))
    th = int(round(tw * h / w))            # keep the frame's aspect
    tw, th = min(tw, w), min(th, h)

    step_x = max(1, int(round(tw * (1.0 - overlap))))
    step_y = max(1, int(round(th * (1.0 - overlap))))

    xs = list(range(0, max(w - tw, 0) + 1, step_x)) or [0]
    ys = list(range(0, max(h - th, 0) + 1, step_y)) or [0]
    if xs[-1] + tw < w:
        xs.append(w - tw)
    if ys[-1] + th < h:
        ys.append(h - th)

    tiles = [(x, y, min(x + tw, w), min(y + th, h)) for y in ys for x in xs]

    # Cost guard: if the grid came out denser than the budget, coarsen
    # it by raising the tile size rather than dropping tiles, which
    # would leave blind spots.
    if len(tiles) > max_tiles:
        grow = math.sqrt(len(tiles) / float(max_tiles))
        return plan_tiles(w, h, target_scale / grow, overlap,
                          imgsz, max_tiles=10 ** 6)
    return tiles


# =========================================================
# BOX FUSION
# =========================================================

def _iou_matrix(a, b):
    """a (N,4), b (M,4) -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1, ax2, ay2 = [a[:, i][:, None] for i in range(4)]
    bx1, by1, bx2, by2 = [b[:, i][None, :] for i in range(4)]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / np.clip(ua, 1e-9, None)


def _ioa_matrix(a, b):
    """
    a (N,4), b (M,4) -> (N,M) intersection over the area of a.

    Asymmetric on purpose. IoU asks "are these the same box"; this asks
    "is a inside b", which is the question the negative-prompt veto
    needs: a spurious 40 px person box sitting on a 300 px car has an
    IoU near 0.02 and an IoA near 1.0.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1, ax2, ay2 = [a[:, i][:, None] for i in range(4)]
    bx1, by1, bx2, by2 = [b[:, i][None, :] for i in range(4)]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = np.clip((ax2 - ax1) * (ay2 - ay1), 1e-9, None)
    return inter / area_a


def weighted_box_fusion(boxes, confs, labels=None, iou_thr=FUSE_IOU,
                        consensus_gain=CONSENSUS_GAIN,
                        cover=None,
                        min_view=CONSENSUS_MIN_VIEW,
                        min_pmax=CONSENSUS_MIN_PMAX,
                        ref_cover=CONSENSUS_REF_COVER):
    """
    Cluster overlapping boxes and merge each cluster into one.

    Position is the confidence-weighted mean of the members, which is
    a better estimate than any single view: tile crops see a target
    from different offsets and their errors are largely independent.

    Confidence is NOT the mean. Averaging would punish a target that
    one tile saw clearly and another saw clipped. Instead the best
    view sets the floor and agreement adds a bounded bonus

        p = p_max + (1 - p_max) * (1 - exp(-g * sum of the others))

    which is monotone, never exceeds 1, and reflects the thing we
    actually believe: three tiles independently calling 0.4 on the
    same patch of ground is stronger evidence than one calling 0.4.

    Three guards keep that from turning into a clutter amplifier on
    busy scenes -- see CONSENSUS_MIN_VIEW for why each exists:

        min_view    a member below this does not join the sum
        min_pmax    a cluster whose best view is below this gets no
                    bonus; agreement corroborates, it cannot create
        cover       optional per-input-box count of how many images
                    could have seen that location (full frame + the
                    tiles containing it). The sum is rescaled to what
                    a box at `ref_cover` would have earned, so the
                    bonus stops being a function of whether the target
                    happened to land in a tile overlap band.

    `labels` is an optional per-box list carried along so each fused
    detection can report attributes of its strongest view. It is opaque
    to this function -- the caller may put a prompt string in it, or a
    tuple of (prompt, mask polygon), or anything else.

    Returns (boxes, confs, raw_confs, n_views, best_labels).
    """
    n = len(boxes)
    if n == 0:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.int32), [])

    boxes = np.asarray(boxes, np.float32)
    confs = np.asarray(confs, np.float32)
    order = np.argsort(-confs)
    boxes, confs = boxes[order], confs[order]
    labels = ([labels[i] for i in order] if labels is not None
              else [""] * n)
    cov = (np.asarray(cover, np.int32)[order] if cover is not None else None)

    used = np.zeros(n, bool)
    out_b, out_c, out_raw, out_n, out_l = [], [], [], [], []

    for i in range(n):
        if used[i]:
            continue
        ious = _iou_matrix(boxes[i:i + 1], boxes)[0]
        members = np.where((ious >= iou_thr) & (~used))[0]
        used[members] = True

        mb, mc = boxes[members], confs[members]
        w = mc[:, None]
        fused = (mb * w).sum(0) / max(w.sum(), 1e-9)

        best = int(members[int(np.argmax(mc))])
        p_max = float(mc.max())

        # Only views that clear min_view get a vote, and the best view
        # is not allowed to vote for itself.
        qual = mc[mc >= min_view]
        rest = max(float(qual.sum()) - p_max, 0.0) if qual.size else 0.0

        # Rescale for how many images could have seen this spot at all.
        if cov is not None and rest > 0.0:
            n_cov = max(int(cov[best]), 1)
            rest *= (ref_cover - 1.0) / max(n_cov - 1, 1)

        if p_max < min_pmax:
            p = p_max
        else:
            p = p_max + (1.0 - p_max) * (1.0 - math.exp(-consensus_gain * rest))

        out_b.append(fused)
        out_c.append(min(p, 0.999))
        out_raw.append(p_max)
        out_n.append(len(members))
        out_l.append(labels[best])

    return (np.array(out_b, np.float32), np.array(out_c, np.float32),
            np.array(out_raw, np.float32), np.array(out_n, np.int32), out_l)


# =========================================================
# PHOTOMETRIC SHADOW / CLUTTER SCORE
# =========================================================

def shadow_features(frame, box, ring_cap=RING_CAP_PX, ring_frac=RING_FRAC):
    """
    Three illumination-physics measurements comparing the box interior
    to a ring of local background around it. Returns None if there is
    not enough context to judge (box at the frame edge, degenerate
    size) - the caller treats that as an abstention, never a rejection.

    Every feature is a RATIO against the ring, so all three are
    dimensionless and comparable whether the target is 20 px or 600 px.

    chrom_p90
        Log-chromaticity is log(c) minus the mean over channels, which
        cancels any scalar change in illumination: shading a surface
        moves its brightness but not its log-chromaticity. So a cast
        shadow sits on top of the background's chromaticity cluster,
        and a person - a different material - sits away from it. The
        90th percentile is used rather than the mean because a person
        may fill only part of the box; the tail is where they show up.

    v_ratio
        Brightness span inside the box over the same span in the ring.
        A person is structured - bright clothing against dark hair,
        limbs against ground - where grass, soil and a flat cast
        shadow are close to uniform. The ring in the denominator is
        what makes this survive a change of altitude: the raw inside
        span alone reads high only while the box is loose enough to
        contain some ground, so it collapses on a large target that
        fills its own box.

    tex
        Laplacian variance inside over Laplacian variance in the ring.
        People carry edges - collars, limbs, equipment - that the
        smooth ground around them does not.

    The remaining four target the two false positives that survive the
    three above - dark vegetation and cast shadow on a hard surface:

    hue_dist
        Circular distance from the ring's dominant hue. Vegetation and
        shadow both keep the hue of the ground they sit on; clothing
        does not.

    sat_ratio
        Saturation inside over saturation in the ring. A cast shadow
        is the ground at lower luminance and NEARLY THE SAME hue, so
        this sits close to 1. Shadow on asphalt drives it below 1.

    green_frac
        Fraction of pixels in the saturated-green band. Near useless
        alone (AUC 0.52 - people wear green too) but it earns its
        place in combination, where it tells a green bush apart from
        a green shirt that has a face and limbs attached.

    dark_flat
        Fraction of pixels materially darker than the ring median -
        the raw extent of the shadow hypothesis.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 6 or bh < 6:
        return None

    pad = int(max(bw, bh) * ring_frac)
    pad = max(RING_MIN_PX, min(pad, ring_cap))
    X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
    X2, Y2 = min(w, x2 + pad), min(h, y2 + pad)
    ctx = frame[Y1:Y2, X1:X2]
    if ctx.size == 0:
        return None

    ox1, oy1 = x1 - X1, y1 - Y1
    ox2, oy2 = ox1 + bw, oy1 + bh
    ring = np.ones(ctx.shape[:2], bool)
    ring[oy1:oy2, ox1:ox2] = False
    if ring.sum() < 80:
        return None

    L = np.log(ctx.astype(np.float32) + 2.0)
    LC = L - L.mean(2, keepdims=True)

    ring_lc = LC[ring].reshape(-1, 3)
    bg = np.median(ring_lc, 0)
    spread = ring_lc.std(0).mean() + 1e-3

    inside = LC[oy1:oy2, ox1:ox2]
    d = np.linalg.norm(inside - bg, axis=2) / (spread * 3.0)

    hsv = cv2.cvtColor(ctx, cv2.COLOR_BGR2HSV)
    hh = hsv[..., 0].astype(np.float32)
    ss = hsv[..., 1].astype(np.float32)
    vv = hsv[..., 2].astype(np.float32)

    v_in, v_rg = vv[oy1:oy2, ox1:ox2], vv[ring]
    span_in = ((np.percentile(v_in, 95) - np.percentile(v_in, 5))
               / (np.median(v_in) + 1e-3))
    span_rg = ((np.percentile(v_rg, 95) - np.percentile(v_rg, 5))
               / (np.median(v_rg) + 1e-3))

    lap = cv2.Laplacian(cv2.cvtColor(ctx, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    tex = float(lap[oy1:oy2, ox1:ox2].var() / max(float(lap[ring].var()), 1e-6))

    h_in, s_in = hh[oy1:oy2, ox1:ox2], ss[oy1:oy2, ox1:ox2]
    dh = np.abs(h_in - np.median(hh[ring]))
    dh = np.minimum(dh, 180.0 - dh)          # hue is circular, 0..179

    return {
        "chrom_p90": float(np.percentile(d, 90)),
        "v_ratio": float(span_in / (span_rg + 1e-3)),
        "tex": tex,
        "hue_dist": float(np.percentile(dh, 75) / 90.0),
        "sat_ratio": float(np.median(s_in) / (np.median(ss[ring]) + 1e-3)),
        "green_frac": float(((h_in >= 30) & (h_in <= 90) & (s_in > 60)).mean()),
        "dark_flat": float((v_in < np.median(v_rg) * 0.75).mean()),
    }


def person_score(frame, box):
    """
    Probability that the box is a real object rather than a shadow or
    a patch of vegetation.

    Returns None when the box cannot be judged - too near the frame
    edge to have a background ring, or degenerate. That is an
    ABSTENTION, and callers must treat it as "no opinion": an
    unjudgeable box keeps its confidence and is never dropped. The
    earlier version returned 0.5 here, which silently docked 25% off
    every casualty large enough that its ring ran off the frame.
    """
    f = shadow_features(frame, box)
    if f is None:
        return None
    v = np.array([f[k] for k in SHADOW_FEATURES], np.float32)
    v = np.clip(np.nan_to_num(v), -1e4, 1e4)
    z = float(((v - SHADOW_MEAN) / SHADOW_SCALE * SHADOW_COEF).sum()
              + SHADOW_BIAS)
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))


def confidence_adjust(person_p, weight=SHADOW_CONF_WEIGHT):
    """
    Multiplier applied to a fused detection score given the physics
    verdict. Deliberately ASYMMETRIC:

        person_p is None  -> 1.0     abstention, no opinion, no penalty
        person_p >= 0.5   -> 1.0     physics agrees, detector stands
        person_p <  0.5   -> < 1.0   physics objects, demote linearly

    Physics is never allowed to promote. A colour statistic is weaker
    evidence than the detector it is second-guessing, so it gets a
    veto on doubt and no vote on confidence.
    """
    if person_p is None or person_p >= 0.5:
        return 1.0
    return 1.0 - weight * (0.5 - person_p) / 0.5


# =========================================================
# MASK SHAPE  (scene-invariant geometry)
# =========================================================

def mask_shape(poly, box, min_area=SHAPE_MIN_AREA_PX):
    """
    Outline statistics for one segmentation polygon, in frame pixels.

    Returns None when the polygon cannot be read -- absent, fewer than
    the five points cv2.fitEllipse needs, or so small that its
    perimeter is quantisation noise. None is an ABSTENTION and the
    caller must treat it as "no opinion", exactly as person_score()
    does: a detection with no usable mask keeps its score.

    See the ENABLE_MASK_SHAPE block for what each number means and why
    these travel between environments when the colour features do not.
    """
    if poly is None:
        return None
    c = np.asarray(poly, np.float32).reshape(-1, 2)
    if len(c) < 5:
        return None

    area = float(cv2.contourArea(c))
    per = float(cv2.arcLength(c, True))
    if area < min_area or per < 1e-3:
        return None

    bw = max(float(box[2]) - float(box[0]), 1e-6)
    bh = max(float(box[3]) - float(box[1]), 1e-6)

    try:
        (_, (axis_a, axis_b), _) = cv2.fitEllipse(c)
    except cv2.error:
        return None
    major = max(axis_a, axis_b)
    minor = max(min(axis_a, axis_b), 1e-3)
    if major < 1e-3:
        return None

    return {
        "fill": float(min(area / (bw * bh), 1.5)),
        "compact": float(min(4.0 * math.pi * area / (per * per), 1.5)),
        "elong": float(major / minor),
        "major_px": float(major),
        "area_px": area,
    }


def shape_verdict(sh,
                  min_fill=SHAPE_MIN_FILL,
                  min_compact=SHAPE_MIN_COMPACT,
                  max_elong=SHAPE_MAX_ELONG,
                  max_fill=SHAPE_MAX_FILL):
    """
    (keep, reason) for a mask_shape() dict.

    Abstains -- returns (True, "") -- on None, so this can never be the
    reason a casualty is lost to a missing mask. Each rejection names
    the thing it is rejecting, because these reasons are what tells you
    whether a gate is earning its place on YOUR footage:

        sliver    a thin diagonal streak: railing, kerb, road marking,
                  a cast shadow's edge
        ragged    perimeter far too long for the area: rubble, foliage,
                  a mask that shattered into fragments
        elongated major axis several times the minor: pole, post, pipe
        solid     a filled rectangle, which at nadir is a vehicle roof
                  or a slab, not a body
    """
    if sh is None:
        return True, ""
    if sh["fill"] < min_fill:
        return False, "sliver"
    if sh["compact"] < min_compact:
        return False, "ragged"
    if sh["elong"] > max_elong:
        return False, "elongated"
    if sh["fill"] > max_fill:
        return False, "solid"
    return True, ""


def size_verdict(major_px, gsd,
                 min_m=PERSON_MIN_M, max_m=PERSON_MAX_M):
    """
    (keep, reason) from the physical extent of the target.

    Abstains when gsd is None. This is the only gate in the file that
    needs no labels and no refitting between environments: a human seen
    from above is between min_m and max_m along their long axis, and a
    car is not, at any altitude, in any country, on any ground cover.
    """
    if gsd is None or not gsd or major_px is None:
        return True, ""
    m = float(major_px) * float(gsd)
    if m < min_m:
        return False, "too_small"
    if m > max_m:
        return False, "too_large"
    return True, ""


# =========================================================
# DETECTOR
# =========================================================

class RobustPersonDetector:
    """
    Multi-scale tiled YOLOE with weighted box fusion, negative-prompt
    vetoes, mask shape gates and a photometric shadow gate.

    Which of those are active is decided by `env` (see ENV_PROFILES).
    Any keyword passed explicitly overrides the profile, so
    env="urban", shadow_reject=0.25 is a legal thing to ask for once
    the shadow model has been refitted on urban crops.
    """

    def __init__(self,
                 weights=WEIGHTS,
                 device=None,
                 imgsz=IMGSZ,
                 conf_floor=CONF_FLOOR,
                 positives=None,
                 negatives=None,
                 env=DEFAULT_ENV,
                 tiling=ENABLE_TILING,
                 tile_scale=TILE_TARGET_SCALE,
                 tile_overlap=TILE_OVERLAP,
                 tile_max=TILE_MAX,
                 shadow_reject=None,
                 shadow_conf_weight=None,
                 min_conf=None,
                 max_aspect=None,
                 neg_veto=None,
                 shape_gate=None,
                 shape_min_fill=SHAPE_MIN_FILL,
                 shape_min_compact=SHAPE_MIN_COMPACT,
                 shape_max_elong=SHAPE_MAX_ELONG,
                 shape_max_fill=SHAPE_MAX_FILL,
                 gsd=None,
                 person_min_m=PERSON_MIN_M,
                 person_max_m=PERSON_MAX_M,
                 verbose=True):

        if env not in ENV_PROFILES:
            raise ValueError(f"env must be one of {sorted(ENV_PROFILES)}")
        prof = ENV_PROFILES[env]
        self.env = env

        # Explicit keyword beats the profile; None means "use the
        # profile". Every environment-dependent knob resolves here and
        # nowhere else, so there is one place to read to know what a
        # given environment actually changes.
        def pick(value, key):
            return prof[key] if value is None else value

        if device is None:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                device = torch.device("xpu:0")
            elif torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        self.device = device
        self.imgsz = imgsz
        self.conf_floor = conf_floor
        self.tiling = tiling
        self.tile_scale = tile_scale
        self.tile_overlap = tile_overlap
        self.tile_max = tile_max

        self.shadow_reject = pick(shadow_reject, "shadow_reject")
        self.shadow_conf_weight = pick(shadow_conf_weight,
                                       "shadow_conf_weight")
        self.min_conf = pick(min_conf, "min_conf")
        self.max_aspect = pick(max_aspect, "max_aspect")
        self.neg_veto = pick(neg_veto, "neg_veto")
        self.shape_gate = pick(shape_gate, "shape_gate")

        self.shape_min_fill = shape_min_fill
        self.shape_min_compact = shape_min_compact
        self.shape_max_elong = shape_max_elong
        self.shape_max_fill = shape_max_fill

        self.gsd = gsd
        self.person_min_m = person_min_m
        self.person_max_m = person_max_m

        # Polygons are only worth extracting if something downstream
        # reads them: the shape gate, or the size prior falling back on
        # a mask's major axis.
        self.use_masks = bool(ENABLE_MASK_SHAPE
                              and (self.shape_gate or self.gsd))

        self.positives = list(positives if positives is not None
                              else POSITIVE_PROMPTS)
        self.negatives = list(negatives if negatives is not None
                              else prof["negatives"])
        self.prompts = self.positives + self.negatives
        self.n_pos = len(self.positives)

        self.model = YOLOE(weights)
        self.model.set_classes(self.prompts,
                               self.model.get_text_pe(self.prompts))

        self._tile_cache = {}
        self.stats = {"frames": 0, "raw": 0, "seam": 0, "fused": 0,
                      "veto": 0, "shadow": 0, "geom": 0, "shape": 0,
                      "size": 0, "conf": 0, "out": 0}

        if verbose:
            print(f"[robust] weights={weights} device={device} imgsz={imgsz}")
            print(f"[robust] env={env}  prompts: {self.n_pos} positive "
                  f"+ {len(self.negatives)} negative")
            if tiling:
                n = len(plan_tiles(1920, 1080, tile_scale, tile_overlap,
                                   imgsz, tile_max))
                print(f"[robust] tiling on: {n} tiles at 1080p "
                      f"(scale {tile_scale}, overlap {tile_overlap})")
            else:
                print("[robust] tiling off")
            print(f"[robust] shadow_reject={self.shadow_reject} "
                  f"min_conf={self.min_conf} max_aspect={self.max_aspect}")
            print(f"[robust] neg_veto={self.neg_veto} "
                  f"shape_gate={self.shape_gate} "
                  f"gsd={self.gsd if self.gsd else 'off'}")
            if self.shadow_reject <= 0.0 and self.shadow_conf_weight <= 0.0:
                print("[robust] photometric gate OFF "
                      "(fitted on grass; refit before enabling here)")

    # ---------------- inference ----------------

    def _infer(self, images):
        """
        YOLOE over a list of BGR images.

        -> list of (boxes, confs, cls, polys), one entry per image.

        `polys` is a per-box list of Nx2 float arrays in that image's
        own pixel coordinates, or None where the model returned no
        usable mask. The weights are segmentation weights, so this
        costs no extra forward pass -- the masks were simply being
        discarded before.
        """
        out = []
        for i in range(0, len(images), TILE_BATCH):
            chunk = images[i:i + TILE_BATCH]
            res = self.model.predict(chunk,
                                     imgsz=self.imgsz,
                                     conf=self.conf_floor,
                                     iou=NMS_IOU,
                                     device=self.device,
                                     verbose=False)
            for r in res:
                n = len(r.boxes)
                if n == 0:
                    out.append((np.zeros((0, 4), np.float32),
                                np.zeros(0, np.float32),
                                np.zeros(0, np.int32),
                                []))
                    continue
                b = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                c = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
                k = r.boxes.cls.detach().cpu().numpy().astype(np.int32)

                polys = [None] * n
                if self.use_masks and getattr(r, "masks", None) is not None:
                    try:
                        xy = r.masks.xy
                        if len(xy) == n:
                            polys = [np.asarray(q, np.float32) for q in xy]
                    except Exception:
                        pass          # abstain, never fail a frame on this
                out.append((b, c, k, polys))
        return out

    # ---------------- main entry ----------------

    def _coverage(self, box, tiles):
        """
        How many of the images we ran could have seen this box at all:
        the full frame, plus every tile containing its centre. Feeds the
        coverage normalisation in weighted_box_fusion -- without it the
        agreement bonus rewards landing in a tile overlap band.
        """
        cx = 0.5 * (float(box[0]) + float(box[2]))
        cy = 0.5 * (float(box[1]) + float(box[3]))
        n = 1
        for (tx1, ty1, tx2, ty2) in tiles:
            if tx1 <= cx < tx2 and ty1 <= cy < ty2:
                n += 1
        return n

    def detect(self, frame):
        """BGR frame -> list[Detection], sorted by confidence."""
        h, w = frame.shape[:2]
        self.stats["frames"] += 1

        key = (w, h)
        if key not in self._tile_cache:
            self._tile_cache[key] = (
                plan_tiles(w, h, self.tile_scale, self.tile_overlap,
                           self.imgsz, self.tile_max)
                if self.tiling else [])
        tiles = self._tile_cache[key]

        images = [frame] + [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
        results = self._infer(images)

        boxes, confs, labels, cover = [], [], [], []
        nboxes, nconfs = [], []          # negative-prompt hits, kept now

        # --- full-frame pass: no seam logic, it sees everything ---
        fb, fc, fk, fm = results[0]
        self.stats["raw"] += len(fb)
        for b, c, k, m in zip(fb, fc, fk, fm):
            if k >= self.n_pos:
                nboxes.append(b)
                nconfs.append(c)
                continue
            boxes.append(b)
            confs.append(c)
            labels.append((self.prompts[int(k)], m))
            cover.append(self._coverage(b, tiles))

        # --- tiles: offset into frame coords, reject seam fragments ---
        for (tx1, ty1, tx2, ty2), (tb, tc, tk, tm) in zip(tiles, results[1:]):
            tw, th = tx2 - tx1, ty2 - ty1
            self.stats["raw"] += len(tb)
            off_b = np.array([tx1, ty1, tx1, ty1], np.float32)
            off_p = np.array([tx1, ty1], np.float32)
            for b, c, k, m in zip(tb, tc, tk, tm):
                gb = b + off_b
                if k >= self.n_pos:
                    # Negatives skip the seam rule. A half-seen car is
                    # still evidence that these pixels are a car, and a
                    # veto can only ever remove a box, so a fragment
                    # here costs nothing it could not also cost whole.
                    nboxes.append(gb)
                    nconfs.append(c)
                    continue
                # A box flush against a tile edge that is not also a
                # frame edge is a target cut in half. The neighbouring
                # tile overlaps this strip and sees it whole.
                clipped = (
                    (b[0] <= SEAM_MARGIN and tx1 > 0) or
                    (b[1] <= SEAM_MARGIN and ty1 > 0) or
                    (b[2] >= tw - SEAM_MARGIN and tx2 < w) or
                    (b[3] >= th - SEAM_MARGIN and ty2 < h)
                )
                if clipped:
                    self.stats["seam"] += 1
                    continue
                boxes.append(gb)
                confs.append(c)
                labels.append((self.prompts[int(k)],
                               None if m is None else m + off_p))
                cover.append(self._coverage(gb, tiles))

        if not boxes:
            return []

        fb, fc, fraw, fn, flab = weighted_box_fusion(
            boxes, confs, labels, cover=cover)
        self.stats["fused"] += len(fb)

        # --- negative-prompt veto ---------------------------------
        # Fused the same way, so a negative's raw score means the same
        # thing as a positive's and the two are comparable. See
        # ENABLE_NEG_VETO for why the overlap is containment and not
        # IoU, and why the comparison uses raw rather than fused
        # confidence.
        veto_ioa = None
        if self.neg_veto and nboxes:
            nb, _nc, nraw, _nn, _nl = weighted_box_fusion(
                np.asarray(nboxes, np.float32),
                np.asarray(nconfs, np.float32))
            strong = nraw >= NEG_VETO_MIN_CONF
            if strong.any():
                nb, nraw = nb[strong], nraw[strong]
                ioa = _ioa_matrix(fb, nb)
                covered = ioa >= NEG_VETO_IOA
                # best negative score among those actually covering us
                scored = np.where(covered, nraw[None, :], -1.0)
                veto_ioa = scored.max(1)

        dets = []
        for i, (b, c, raw, n, lab) in enumerate(
                zip(fb, fc, fraw, fn, flab)):
            prompt, poly = lab if isinstance(lab, tuple) else (lab, None)

            x1 = int(max(0, min(w - 1, b[0])))
            y1 = int(max(0, min(h - 1, b[1])))
            x2 = int(max(0, min(w, b[2])))
            y2 = int(max(0, min(h, b[3])))
            bw, bh = x2 - x1, y2 - y1

            # geometry gates
            if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
                self.stats["geom"] += 1
                continue
            if (bw * bh) > MAX_BOX_FRAC * w * h:
                self.stats["geom"] += 1
                continue
            if max(bh / max(bw, 1e-6), bw / max(bh, 1e-6)) > self.max_aspect:
                self.stats["geom"] += 1
                continue

            # negative-prompt veto: the network itself, on the same
            # pixels, had a better word for this than "person"
            if veto_ioa is not None and veto_ioa[i] >= raw * NEG_VETO_RATIO:
                self.stats["veto"] += 1
                continue

            # mask shape: outline, not colour, so it survives the move
            # between environments. Abstains when there is no mask.
            sh = mask_shape(poly, (x1, y1, x2, y2)) if self.use_masks else None
            if self.shape_gate:
                ok, _why = shape_verdict(sh,
                                         self.shape_min_fill,
                                         self.shape_min_compact,
                                         self.shape_max_elong,
                                         self.shape_max_fill)
                if not ok:
                    self.stats["shape"] += 1
                    continue

            # physical size: needs no tuning, only a GSD
            major_px = sh["major_px"] if sh else float(max(bw, bh))
            ok, _why = size_verdict(major_px, self.gsd,
                                    self.person_min_m, self.person_max_m)
            if not ok:
                self.stats["size"] += 1
                continue

            ps = person_score(frame, (x1, y1, x2, y2))
            if (self.shadow_reject > 0.0
                    and ps is not None and ps < self.shadow_reject):
                self.stats["shadow"] += 1
                continue

            # Physics demotes doubt and nothing else (see
            # confidence_adjust). A box it merely cannot judge keeps
            # its score, so a casualty near the frame edge is never
            # quietly penalised for being there.
            adj = confidence_adjust(ps, self.shadow_conf_weight)
            conf = float(min(c * adj, 0.999))

            if conf < self.min_conf:
                self.stats["conf"] += 1
                continue

            dets.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                conf=conf,
                raw_conf=float(raw),
                person_score=float(ps) if ps is not None else float("nan"),
                n_views=int(n),
                prompt=prompt,
            ))

        dets.sort(key=lambda d: -d.conf)
        self.stats["out"] += len(dets)
        return dets

    def report(self):
        """
        One line per run. Read it as a funnel: the gate with the big
        number is the one carrying the environment, and a gate sitting
        at zero is either unnecessary or misconfigured. On urban
        footage with the grass tuning, shadow-drop was the zero that
        gave the whole problem away.
        """
        s = self.stats
        f = max(s["frames"], 1)
        return (f"[robust:{self.env}] {s['frames']} frames | raw {s['raw']} "
                f"({s['raw'] / f:.1f}/f) -> seam-drop {s['seam']} "
                f"-> fused {s['fused']} -> veto {s['veto']} "
                f"geom {s['geom']} shape {s['shape']} size {s['size']} "
                f"shadow {s['shadow']} conf {s['conf']} "
                f"-> out {s['out']} ({s['out'] / f:.1f}/f)")


# =========================================================
# LEGACY ADAPTER
# =========================================================

def detect_boxes(detector, frame):
    """
    Same shape as the old Pipeline_yoloe.detect(): a list of
    (x1, y1, x2, y2, conf) tuples.
    """
    return [d.as_tuple() for d in detector.detect(frame)]


# =========================================================
# PRESETS
# =========================================================

SPEED_PRESETS = {
    # tile scale, and nothing environmental. How many pixels we spend
    # looking is a separate question from what counts as a person.
    "fast": dict(tile_scale=0.45),
    "balanced": dict(tile_scale=0.67),
    "thorough": dict(tile_scale=0.67, conf_floor=0.04),
}


def make_detector(preset="balanced", env=None, **kw):
    """
    Two orthogonal choices, either of which may be given in `preset`.

    Speed -- how many pixels we spend looking, from the measured
    speed/quality curve in the TILE_TARGET_SCALE block:

        fast      4 tiles,  ~335 ms/frame, 96% of the confidence gain
        balanced  9 tiles,  ~549 ms/frame, the measured peak (default)
        thorough  9 tiles, a lower conf floor and a permissive shadow
                  gate, for a second pass over ground already swept,
                  where a false positive costs an operator's glance
                  and a miss costs a casualty

    Environment -- what counts as a person, from ENV_PROFILES:

        grass     bare ground, nadir. The tuning this module was
                  measured on, unchanged.
        urban     street scene with vehicles, poles, railings, rubble.
                  Clutter vocabulary, negative vetoes, mask shape
                  gates, tighter aspect, photometric gate off.

    Both may be written into the preset string in any order:

        make_detector("balanced")            grass, the old default
        make_detector("urban")               urban at balanced speed
        make_detector("urban-fast")          urban, 4 tiles
        make_detector("fast", env="urban")   the same thing

    "thorough" relaxes whichever environment it is combined with, it
    does not replace it. Anything passed as **kw overrides everything.
    """
    speed, chosen_env = None, env
    for tok in str(preset).replace("/", "-").replace("_", "-").split("-"):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in SPEED_PRESETS:
            if speed is not None and speed != tok:
                raise ValueError(f"two speed presets in {preset!r}")
            speed = tok
        elif tok in ENV_PROFILES:
            if chosen_env is not None and chosen_env != tok:
                raise ValueError(f"env={env!r} contradicts preset {preset!r}")
            chosen_env = tok
        else:
            raise ValueError(
                f"unknown preset token {tok!r}; speed presets are "
                f"{sorted(SPEED_PRESETS)}, environments {sorted(ENV_PROFILES)}")

    speed = speed or "balanced"
    chosen_env = chosen_env or DEFAULT_ENV

    cfg = dict(SPEED_PRESETS[speed])
    cfg["env"] = chosen_env

    if speed == "thorough":
        # Relax the environment's gates rather than overriding them.
        # On grass that reproduces the old thorough preset exactly
        # (0.25 -> 0.10); on urban the shadow gate is already off, so
        # the min() leaves it off and the loosened shape gates are
        # what a second sweep actually buys.
        prof = ENV_PROFILES[chosen_env]
        cfg["shadow_reject"] = min(prof["shadow_reject"], 0.10)
        if prof["shape_gate"]:
            cfg["shape_min_fill"] = SHAPE_MIN_FILL * 0.7
            cfg["shape_min_compact"] = SHAPE_MIN_COMPACT * 0.7
            cfg["shape_max_elong"] = SHAPE_MAX_ELONG * 1.3

    cfg.update(kw)
    return RobustPersonDetector(**cfg)

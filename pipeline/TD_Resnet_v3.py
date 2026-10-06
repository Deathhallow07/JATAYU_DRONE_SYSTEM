"""
TD_ResNet v8 — Triage Re-ID
============================================
Fixes in this revision
──────────────────────
1. BN-before-Linear crash (previous fix retained):
   bottleneck Linear(2048→512) is applied BEFORE bn_bottle SyncBatchNorm(512).

2. LR schedule bug — lr_head at p=0 (end of warmup) returned 0.6, not 1.0,
   causing an immediate 40% LR drop right after warmup ended.
   Fix: lr_head now uses the same clean cosine formula as lr_backbone,
   reaching exactly 1.0 at the warmup boundary and decaying smoothly to 0.01.
   Old formula:  0.5*(1+cos(π·p))*0.5 + 0.1  →  broken (0.6 at p=0)
   New formula:  0.5*(1+cos(π·p))             →  correct (1.0 at p=0)

3. CE loss stagnation:
   a) CE weight is ramped from 0.1 → CE_WEIGHT over the first WARMUP_EPOCHS
      so the backbone stabilises before the classifier is pushed hard.
   b) Classifier weight_decay lowered from 5e-4 → 1e-4; a linear head on a
      ~200-ID dataset over-regularises quickly at the backbone rate, which
      suppresses the cross-entropy gradient signal.

4. Circle loss spike reduction — all CACHE_SIZE micro-batches are now used
   as anchors in sequence (each with its own forward + partial backward) and
   their loss contributions are averaged before the single optimizer step.
   Memory cost is unchanged: only one live autograd graph exists at a time.
   This eliminates the ±5 epoch-to-epoch swings caused by picking a single
   random anchor per GradCache cycle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
import os
import PIL.Image as Image
import random
import math
from collections import defaultdict
from tqdm import tqdm
import copy

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
"""
================ FINAL METRICS ================

TP: 5
TN: 54
FP: 2
FN: 3
Total decisions: 64

Accuracy : 0.9219
Precision: 0.7143
Recall   : 0.6250

==============================================

[DONE]
0.921875 22
================ FINAL METRICS ================

TP: 4
TN: 56
FP: 0
FN: 4
Total decisions: 64

Accuracy : 0.9375
Precision: 1.0000
Recall   : 0.5000

==============================================

[DONE]
0.9375 3
[]
None
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMAGE_HEIGHT = 256
IMAGE_WIDTH  = 128

# ── MEMORY BUDGET ─────────────────────────────────────────────────────────────
P_MICRO       = 3
K_MICRO       = 2
BATCH_PER_GPU = P_MICRO * K_MICRO    # = 6

CACHE_SIZE    = 4

NUM_GPUS      = 2
EPOCHS        = 200
WARMUP_EPOCHS = 15
SWA_START     = 160
MARGIN        = 0.35

CE_WEIGHT     = 0.5
CIRCLE_SCALE  = 64
CIRCLE_MARGIN = 0.35

DATASET_PATH = "/home/uasdtu/Documents/Tanmay/prai_for_td"
OUTPUT_DIR   = "/home/uasdtu/Documents/Tanmay/TD/TD_ResNet/v8_checkpoints"

RESUME_CHECKPOINT = None

NUM_STRIPES = 6

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
transform_train = T.Compose([
    T.Resize((288, 144)),
    T.RandomCrop((256, 128)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.08),
    T.RandomGrayscale(p=0.2),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.6, scale=(0.05, 0.30), ratio=(0.3, 3.0), value=0),
    T.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0),
])

transform_val = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── DATASET ───────────────────────────────────────────────────────────────────
class ReIDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform      = transform
        self.samples        = []
        self.pid_to_indices = defaultdict(list)

        pid_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.num_pids = 0
        for pid_idx, pid_dir in enumerate(pid_dirs):
            pid_path = os.path.join(root_dir, pid_dir)
            images   = [
                f for f in os.listdir(pid_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if len(images) < 2:
                continue
            for img_name in sorted(images):
                path = os.path.join(pid_path, img_name)
                self.samples.append((path, self.num_pids))
                self.pid_to_indices[self.num_pids].append(len(self.samples) - 1)
            self.num_pids += 1

        self.pids = list(self.pid_to_indices.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(pid, dtype=torch.long)


# ── PK SAMPLER ────────────────────────────────────────────────────────────────
class PKSamplerDDP(Sampler):
    def __init__(self, dataset, p, k,
                 num_batches_per_epoch=512,
                 rank=0, num_replicas=2):
        self.dataset     = dataset
        self.p           = p
        self.k           = k
        self.num_batches = num_batches_per_epoch
        self.rank        = rank
        self.pids        = dataset.pids
        self.epoch       = 0

    def set_epoch(self, e): self.epoch = e
    def __len__(self):      return self.num_batches

    def __iter__(self):
        rng      = random.Random(self.epoch * 31337 + self.rank * 7919)
        all_pids = self.pids.copy()
        rng.shuffle(all_pids)
        qpos = 0

        for _ in range(self.num_batches):
            if qpos + self.p > len(all_pids):
                rng.shuffle(all_pids)
                qpos = 0

            sel_pids = all_pids[qpos: qpos + self.p]
            qpos    += self.p

            batch = []
            for pid in sel_pids:
                idxs = self.dataset.pid_to_indices[pid]
                chosen = rng.sample(idxs, self.k) if len(idxs) >= self.k \
                    else rng.choices(idxs, k=self.k)
                batch.extend(chosen)
            yield batch


# ── LOSSES ────────────────────────────────────────────────────────────────────
def all_gather_cat(tensor: torch.Tensor) -> torch.Tensor:
    ws = dist.get_world_size()
    if ws == 1:
        return tensor
    buf = [torch.zeros_like(tensor) for _ in range(ws)]
    dist.all_gather(buf, tensor)
    buf[dist.get_rank()] = tensor
    return torch.cat(buf, dim=0)


class CircleLoss(nn.Module):
    """Circle Loss — fully vectorized."""
    def __init__(self, s: float = 64, m: float = 0.35):
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, embeddings, labels):
        sim = embeddings @ embeddings.T
        lc       = labels.unsqueeze(1)
        pos_mask = (lc == lc.T).float()
        pos_mask.fill_diagonal_(0)
        neg_mask = 1.0 - (lc == lc.T).float()

        alpha_p = torch.clamp(1 + self.m - sim.detach(), min=0) * pos_mask
        alpha_n = torch.clamp(sim.detach() + self.m,     min=0) * neg_mask

        delta_p = 1 - self.m
        delta_n = self.m

        BIG = 1e4
        lp  = self.s * alpha_p * (sim - delta_p) - (1 - pos_mask) * BIG
        ln  = self.s * alpha_n * (sim - delta_n) - (1 - neg_mask) * BIG

        has_pos = pos_mask.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return embeddings.sum() * 0.0

        loss = torch.log(1 +
                         torch.exp(torch.logsumexp(lp, dim=1)) *
                         torch.exp(torch.logsumexp(ln, dim=1)))
        return loss[has_pos].mean()


class OnlineHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        B        = embeddings.size(0)
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        lc       = labels.view(-1, 1)
        pos_mask = (lc == lc.T).float() - torch.eye(B, device=embeddings.device)
        neg_mask = (lc != lc.T).float()

        valid = pos_mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return embeddings.sum() * 0.0, 0.0

        hp = (dist_mat * pos_mask).max(dim=1).values
        hn = (dist_mat + (1 - neg_mask) * 1e9).min(dim=1).values
        loss = F.softplus(hp - hn)
        active = (loss[valid] > 1e-6).float().mean().item()
        return loss[valid].mean(), active


# ── SPATIAL ATTENTION ─────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


# ── TD CONVOLVER BLOCK ────────────────────────────────────────────────────────
class TD_CONVOLVER_BLOCK(nn.Module):
    def __init__(self, filter_size, in_h, in_w, padding,
                 input_channel, number_of_filters):
        super().__init__()
        F_ = number_of_filters
        C  = input_channel
        k  = filter_size

        self.in_h    = in_h
        self.in_w    = in_w
        self.padding = padding
        self.F       = F_
        self.C       = C
        self.k       = k

        self.w1           = nn.Parameter(torch.empty(F_, C, k, in_h))
        self.w2           = nn.Parameter(torch.empty(F_, C, in_w, k))
        self.phi          = nn.Parameter(torch.zeros(F_, C, k, k))
        self.blend_logits = nn.Parameter(torch.zeros(F_, 2))
        self.alpha_gate   = nn.Parameter(torch.tensor(-3.0))

        self.norm = nn.SyncBatchNorm(F_)

        bottleneck   = max(16, F_ // 4)
        self.se_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se_fc1  = nn.Linear(F_, bottleneck)
        self.se_fc2  = nn.Linear(bottleneck, F_)
        self.spatial = SpatialAttention(kernel_size=7)

        nn.init.kaiming_normal_(self.w1, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.w2, mode='fan_out', nonlinearity='relu')

    def forward(self, x, last_layer_filter=None):
        B     = x.shape[0]
        x_exp = x.unsqueeze(1).expand(B, self.F, self.C, self.in_h, self.in_w)

        a1 = torch.einsum('fckh,bfchw->bfckw', self.w1, x_exp)
        a2 = torch.einsum('bfckw,fcwj->bfckj', a1, self.w2)

        si    = a2.reshape(B * self.F, self.C, self.k, self.k)
        blend = torch.softmax(self.blend_logits, dim=1)
        wd    = blend[:, 0].view(self.F, 1, 1, 1)
        ws    = blend[:, 1].view(self.F, 1, 1, 1)
        si_r  = si.reshape(B, self.F, self.C, self.k, self.k)
        alpha = (wd * si_r + ws * self.phi).reshape(
            B * self.F, self.C, self.k, self.k)

        gate = torch.sigmoid(self.alpha_gate)
        if last_layer_filter is not None:
            alpha = alpha + gate * last_layer_filter
        else:
            alpha = alpha + gate * alpha.detach() * 0.0

        x_in = x.reshape(1, B * self.C, self.in_h, self.in_w)
        zi   = F.conv2d(x_in, alpha, padding=self.padding, groups=B)
        zi   = zi.reshape(B, self.F, self.in_h, self.in_w)
        zi   = F.relu(self.norm(zi))

        se  = torch.sigmoid(self.se_fc2(F.relu(
            self.se_fc1(self.se_pool(zi).flatten(1)))))
        zi  = zi * se.view(B, self.F, 1, 1)
        zi  = self.spatial(zi)
        return zi, alpha


# ── PART-AWARE POOLING ────────────────────────────────────────────────────────
class HorizontalStripePool(nn.Module):
    def __init__(self, num_stripes: int, feat_dim: int, embed_dim: int):
        super().__init__()
        self.S     = num_stripes
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)) for _ in range(num_stripes)
        ])
        self.projs = nn.ModuleList([
            nn.Linear(feat_dim, embed_dim) for _ in range(num_stripes)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        stripe_h   = H // self.S
        parts      = []
        for i, (pool, proj) in enumerate(zip(self.pools, self.projs)):
            h_start = i * stripe_h
            h_end   = (i + 1) * stripe_h if i < self.S - 1 else H
            stripe  = x[:, :, h_start:h_end, :]
            parts.append(proj(pool(stripe).flatten(1)))
        return parts


# ── NETWORK V8 ────────────────────────────────────────────────────────────────
class TD_2_NETWORK_V8(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.td1   = TD_CONVOLVER_BLOCK(3, 256, 128, 1, 3,   128)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.td2   = TD_CONVOLVER_BLOCK(3, 128,  64, 1, 128, 128)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.td3   = TD_CONVOLVER_BLOCK(3,  64,  32, 1, 128, 128)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.td4   = TD_CONVOLVER_BLOCK(3,  32,  16, 1, 128, 128)
        self.td5   = TD_CONVOLVER_BLOCK(3,  32,  16, 1, 128, 128)

        self.avg   = nn.AdaptiveAvgPool2d((1, 1))
        self.proj1 = nn.Linear(128, 256)
        self.proj2 = nn.Linear(128, 256)
        self.proj3 = nn.Linear(128, 256)
        self.proj4 = nn.Linear(128, 256)
        self.proj5 = nn.Linear(128, 256)

        self.stripe_pool = HorizontalStripePool(
            num_stripes=NUM_STRIPES, feat_dim=128, embed_dim=128)

        RAW_DIM   = 256 * 5 + 128 * NUM_STRIPES   # 1280 + 768 = 2048
        EMBED_DIM = 512

        # FIX 1: Linear first, BN second — shapes agree on the backward pass
        self.bottleneck = nn.Linear(RAW_DIM, EMBED_DIM, bias=False)
        self.bn_bottle  = nn.SyncBatchNorm(EMBED_DIM)

        self.bn_neck    = nn.SyncBatchNorm(EMBED_DIM)
        self.bn_neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(EMBED_DIM, num_classes, bias=False)

        self._embed_dim = EMBED_DIM

    @property
    def embed_dim(self):
        return self._embed_dim

    def forward(self, x, return_logits: bool = False):
        z1, a1 = self.td1(x);           z1p = self.pool1(z1)
        z2, a2 = self.td2(z1p);         z2p = self.pool2(z2)
        z3, a3 = self.td3(z2p, a2);     z3p = self.pool3(z3)
        z4, a4 = self.td4(z3p, a3)
        z5, _  = self.td5(z4,  a4)

        g = torch.cat([
            self.proj1(self.avg(z1).flatten(1)),
            self.proj2(self.avg(z2).flatten(1)),
            self.proj3(self.avg(z3).flatten(1)),
            self.proj4(self.avg(z4).flatten(1)),
            self.proj5(self.avg(z5).flatten(1)),
        ], dim=1)                                         # (B, 1280)

        parts = self.stripe_pool(z5)
        p     = torch.cat(parts, dim=1)                  # (B, 768)

        feat  = torch.cat([g, p], dim=1)                 # (B, 2048)
        feat  = self.bottleneck(feat)                     # (B, 512)  Linear first
        feat  = self.bn_bottle(feat)                      # SyncBN(512) then BN
        neck  = self.bn_neck(feat)
        emb   = F.normalize(neck, p=2, dim=1)

        if return_logits:
            return emb, self.classifier(neck)
   
        return emb


# ── EMA HELPER ────────────────────────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model, decay=0.9997):
        self.ema   = copy.deepcopy(model.module).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, m_p in zip(self.ema.parameters(),
                               model.module.parameters()):
            ema_p.data.mul_(self.decay).add_(m_p.data, alpha=1 - self.decay)


# ── CHECKPOINT ────────────────────────────────────────────────────────────────
def load_checkpoint(path, model, optimizer, scheduler, scaler, rank):
    if rank == 0:
        print(f"\n[Resume] {path}")
    ckpt = torch.load(path, map_location=f"cuda:{rank}")
    sd   = ckpt["model_state_dict"]
    if not any(k.startswith("module.") for k in sd):
        sd = {"module." + k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    for key, obj in [("optimizer_state_dict", optimizer),
                     ("scheduler_state_dict", scheduler),
                     ("scaler_state_dict",    scaler)]:
        if ckpt.get(key):
            try:
                obj.load_state_dict(ckpt[key])
            except Exception:
                pass
    ep = ckpt.get("epoch", 0)
    if rank == 0:
        print(f"[Resume] epoch={ep}\n")
    return ep


# ── TRAINING WORKER ───────────────────────────────────────────────────────────
def train(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = ReIDDataset(DATASET_PATH, transform=transform_train)
    if rank == 0:
        pool = BATCH_PER_GPU * NUM_GPUS * CACHE_SIZE
        ids  = P_MICRO * NUM_GPUS * CACHE_SIZE
        print(f"\n{'='*60}")
        print(f"  TD_ResNet v8 — Triage ReID")
        print(f"{'='*60}")
        print(f"  Dataset : {len(dataset)} images | {dataset.num_pids} IDs")
        print(f"  Batch   : P={P_MICRO} K={K_MICRO} × {NUM_GPUS} GPUs = "
              f"{BATCH_PER_GPU * NUM_GPUS} imgs/step")
        print(f"  Pool    : {pool} embeddings (~{ids} unique IDs)")
        print(f"  Max simultaneous triplets: "
              f"{P_MICRO * NUM_GPUS * CACHE_SIZE // 3} "
              f"(target ≤4 per gradient step)")
        print(f"{'='*60}\n")

    sampler = PKSamplerDDP(dataset, p=P_MICRO, k=K_MICRO,
                           num_batches_per_epoch=512,
                           rank=rank, num_replicas=world_size)

    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=4, pin_memory=True,
                        persistent_workers=True)

    model = TD_2_NETWORK_V8(num_classes=dataset.num_pids).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    ema = ModelEMA(model, decay=0.9997) if rank == 0 else None

    circle_crit  = CircleLoss(s=CIRCLE_SCALE, m=CIRCLE_MARGIN)
    triplet_crit = OnlineHardTripletLoss(margin=MARGIN)
    ce_crit      = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ── Param groups ──────────────────────────────────────────────────────────
    backbone_params = list(model.module.td1.parameters()) + \
                      list(model.module.td2.parameters()) + \
                      list(model.module.td3.parameters()) + \
                      list(model.module.td4.parameters()) + \
                      list(model.module.td5.parameters())
    classifier_params = list(model.module.classifier.parameters())
    head_params = [p for p in model.parameters()
                   if not any(p is bp for bp in backbone_params)
                   and not any(p is cp for cp in classifier_params)]

    LR_BACKBONE   = 3e-4
    LR_HEAD       = 6e-4
    LR_CLASSIFIER = 1e-3

    optimizer = torch.optim.AdamW([
        {"params": backbone_params,   "lr": LR_BACKBONE,   "weight_decay": 5e-4},
        {"params": head_params,       "lr": LR_HEAD,       "weight_decay": 5e-4},
        # FIX 3b: lower weight_decay for classifier — a linear head on a small
        # dataset (~200 IDs) over-regularises at the backbone rate, killing the
        # CE gradient signal that drives identity learning.
        {"params": classifier_params, "lr": LR_CLASSIFIER, "weight_decay": 1e-4},
    ])

    # ── LR schedules (FIX 2) ─────────────────────────────────────────────────
    # Old lr_head: 0.5*(1+cos(π·p))*0.5 + 0.1  → evaluates to 0.6 at p=0
    #              (immediately after warmup) instead of 1.0.
    # New lr_head: 0.5*(1+cos(π·p))             → 1.0 at p=0, smooth decay.
    # Both backbone and head now share the same cosine shape.

    def lr_backbone(ep):
        if ep < WARMUP_EPOCHS:
            return (ep + 1) / WARMUP_EPOCHS
        if ep >= SWA_START:
            return 0.02
        p = (ep - WARMUP_EPOCHS) / max(1, SWA_START - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * p))

    def lr_head(ep):
        if ep < WARMUP_EPOCHS:
            return (ep + 1) / WARMUP_EPOCHS
        if ep >= SWA_START:
            return 0.01
        p = (ep - WARMUP_EPOCHS) / max(1, SWA_START - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * p))   # FIX: was *0.5+0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lr_backbone, lr_head, lr_head])
    scaler    = torch.cuda.amp.GradScaler()

    swa_model = torch.optim.swa_utils.AveragedModel(model) if rank == 0 else None
    swa_sched  = torch.optim.swa_utils.SWALR(
        optimizer, swa_lr=1e-5, anneal_epochs=5) if rank == 0 else None

    start_epoch = 0
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        start_epoch = load_checkpoint(
            RESUME_CHECKPOINT, model, optimizer, scheduler, scaler, rank)

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS):
        sampler.set_epoch(epoch)
        model.train()

        if epoch == 60 and rank == 0:
            print("[Info] Raising label smoothing to 0.1")
        ce_crit = nn.CrossEntropyLoss(
            label_smoothing=0.1 if epoch >= 60 else 0.05)

        # FIX 3a: ramp CE weight 0.1 → CE_WEIGHT over warmup epochs so the
        # backbone can stabilise before the classifier is pushed hard.
        ce_w = CE_WEIGHT if epoch >= WARMUP_EPOCHS else \
            0.1 + (CE_WEIGHT - 0.1) * (epoch / WARMUP_EPOCHS)

        run_trip = 0.0; run_circ = 0.0; run_ce = 0.0
        updates  = 0;   active_fracs = []

        optimizer.zero_grad()
        pbar      = tqdm(loader,
                         desc=f"[GPU {rank}] Epoch {epoch+1}/{EPOCHS}",
                         disable=(rank != 0))
        it        = iter(pbar)
        exhausted = False

        while not exhausted:
            # ── Phase 1: fill GradCache (no-grad) ─────────────────────────
            cache_imgs  = []
            cache_pids  = []
            cache_embs  = []
            steps_taken = 0

            for _ in range(CACHE_SIZE):
                try:
                    imgs, pids = next(it)
                except StopIteration:
                    exhausted = True
                    break

                imgs = imgs.to(rank, non_blocking=True)
                pids = pids.to(rank, non_blocking=True)

                with torch.no_grad(), torch.cuda.amp.autocast():
                    emb_sg = model(imgs)

                cache_imgs.append(imgs)
                cache_pids.append(pids)
                cache_embs.append(emb_sg.detach())
                steps_taken += 1

            if steps_taken == 0:
                break

            # ── Phase 2: gradient update ───────────────────────────────────
            # FIX 4: iterate ALL cached micro-batches as anchors instead of
            # picking one at random.  Each anchor gets its own no-grad pool
            # (only itself is live) so memory cost is identical.  Gradients
            # accumulate across the loop and the optimiser step fires once at
            # the end, averaging over all CACHE_SIZE contributions.  This
            # eliminates the high epoch-to-epoch variance in circle loss.

            acc_circ = torch.zeros(1, device=rank)
            acc_trip = torch.zeros(1, device=rank)
            acc_ce   = torch.zeros(1, device=rank)

            for anchor_idx in range(steps_taken):
                with model.no_sync():
                    with torch.cuda.amp.autocast():
                        emb_grad, logits_grad = model(
                            cache_imgs[anchor_idx], return_logits=True)

                pool_embs             = list(cache_embs)
                pool_embs[anchor_idx] = emb_grad

                all_local_embs   = torch.cat(pool_embs,  dim=0)
                all_local_labels = torch.cat(cache_pids, dim=0)

                all_embs   = all_gather_cat(all_local_embs)
                all_labels = all_gather_cat(all_local_labels)

                with torch.cuda.amp.autocast():
                    circ_loss           = circle_crit(all_embs, all_labels)
                    trip_loss, act_frac = triplet_crit(all_embs, all_labels)
                    ce_loss             = ce_crit(logits_grad,
                                                  cache_pids[anchor_idx])
                    # Divide by steps_taken so total gradient magnitude is
                    # independent of how many anchors were in this cache cycle.
                    step_loss = (circ_loss
                                 + 0.3 * trip_loss
                                 + ce_w * ce_loss) / steps_taken

                acc_circ += circ_loss.detach() / steps_taken
                acc_trip += trip_loss.detach() / steps_taken
                acc_ce   += ce_loss.detach()   / steps_taken
                active_fracs.append(act_frac)

                scaler.scale(step_loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if rank == 0 and ema is not None:
                ema.update(model)

            run_circ += acc_circ.item()
            run_trip += acc_trip.item()
            run_ce   += acc_ce.item()
            updates  += 1

            if rank == 0:
                n_ = max(updates, 1)
                pbar.set_postfix({
                    "circ":     f"{acc_circ.item():.3f}",
                    "trip":     f"{acc_trip.item():.3f}",
                    "ce":       f"{acc_ce.item():.3f}",
                    "avg_circ": f"{run_circ / n_:.3f}",
                    "avg_trip": f"{run_trip / n_:.3f}",
                    "avg_ce":   f"{run_ce   / n_:.3f}",
                    "act":      f"{sum(active_fracs[-steps_taken:]) / steps_taken:.0%}",
                    "lr_bb":    f"{optimizer.param_groups[0]['lr']:.1e}",
                    "lr_cls":   f"{optimizer.param_groups[2]['lr']:.1e}",
                    "ce_w":     f"{ce_w:.2f}",
                })

        # SWA / scheduler step
        if rank == 0 and epoch >= SWA_START and swa_model is not None:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            scheduler.step()

        if rank == 0:
            n    = max(updates, 1)
            path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1:03d}.pth")
            ckpt = {
                "epoch":                epoch + 1,
                "model_state_dict":     model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "loss_circle":          run_circ / n,
                "loss_triplet":         run_trip / n,
                "loss_ce":              run_ce   / n,
            }
            if ema is not None:
                ckpt["ema_state_dict"] = ema.ema.state_dict()
            torch.save(ckpt, path)

            ma = sum(active_fracs) / max(len(active_fracs), 1)
            print(f"\n[Epoch {epoch+1:03d}]  "
                  f"circ={run_circ/n:.4f}  "
                  f"trip={run_trip/n:.4f}  "
                  f"ce={run_ce/n:.4f}  "
                  f"ce_w={ce_w:.2f}  "
                  f"active={ma:.1%}  "
                  f"lr_bb={optimizer.param_groups[0]['lr']:.2e}  "
                  f"lr_head={optimizer.param_groups[1]['lr']:.2e}  "
                  f"lr_cls={optimizer.param_groups[2]['lr']:.2e}")
            print("-" * 60)

    # Final SWA BN update
    if rank == 0 and swa_model is not None:
        val_loader = DataLoader(
            ReIDDataset(DATASET_PATH, transform=transform_val),
            batch_size=32, shuffle=False, num_workers=4)
        torch.optim.swa_utils.update_bn(val_loader, swa_model, device=rank)
        torch.save({"swa_model_state_dict": swa_model.module.state_dict()},
                   os.path.join(OUTPUT_DIR, "swa_final.pth"))
        print("\n[Done] SWA model saved → swa_final.pth")

    dist.destroy_process_group()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.spawn(train, args=(NUM_GPUS,), nprocs=NUM_GPUS, join=True)
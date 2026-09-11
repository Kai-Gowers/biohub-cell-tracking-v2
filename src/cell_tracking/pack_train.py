"""Training the pack's way: TemporalUNet3D + node transformer, end to end (ported).

Port of ``scripts/train_unet_transformer.py`` from the pilkwang support pack
(RoyerLab Kaggle baseline), retrieved 2026-09-11. The optimisation is kept
verbatim -- AdamW at ``lr`` (torch default weight decay), no scheduler,
``clip_grad_norm_(1.0)``, ``loss = edge_loss + det_loss_weight * det_loss``,
gradient checkpointing, full-frame 2-frame windows, brightness shift +
independent z/y/x flips, detect-and-match edge targets from the live
detector's peaks (logit > 0.3, 5 µm pool kernel, 5 µm greedy match), the
focal-weighted softmax-over-sources BCE on annotated rows/columns, and the
one-hot weighted-BCE detection loss (``w_pos = 1/n_pos``, ``w_neg = 0.01/n_neg``).

What is added around it, none of which changes the gradient:

* a resumable checkpoint (``state_dict`` + optimizer + epoch + history),
  ``--save-every`` snapshots and a clean ``max_hours`` stop for SLURM chains;
* ``--seed`` that also seeds the augmentation RNG (the original drew fresh
  OS entropy per item);
* every ``eval_tracking_every`` epochs, the real inference path
  (``pipeline.run_volume`` at stage ``ilp``, no TTA, no post-processing) on
  the held-out volumes scored with the competition metric -> ``val_score``
  (+ per-embryo), which selects ``<out>_best.pt``. The pack's own
  ``acc x node-recall`` validation is still computed and logged as
  ``pack_val_*`` but does not select.

Data come from the decimated-frame cache (``cache.VolumeFrames``) or straight
from the zarr; ground truth from ``io_geff.read_geff``.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cell_tracking.cache import VolumeFrames, cache_path
from cell_tracking.config import DOWNSAMPLE, VAL_FRAC, VAL_SEED
from cell_tracking.evaluate import score_volume_graphs, summarize, summarize_by_embryo, track_graph_to_geff
from cell_tracking.io_geff import list_geff_datasets, read_geff, split_dataset_names
from cell_tracking.models.unet_node_transformer import (
    _POS_EMBED_DIM,
    DEFAULT_MODEL_CONFIG,
    UNetNodeTransformer,
    _pos_embed_torch,
    build_model,
    normalize_state_dict,
)


class TrainingDivergedError(RuntimeError):
    pass


@dataclass
class TrainConfig:
    """The seed-314159 recipe (``training_config.json``) plus our additions."""

    epochs: int = 400
    lr: float = 1e-4
    batch_size: int = 8
    num_workers: int = 4
    window_size: int = 2
    unet_out_channels: int = 32
    unet_layers: tuple[int, ...] = (32, 64, 128)
    det_loss_weight: float = 1.0
    det_neg_weight: float = 0.01
    pool_kernel_um: float = 5.0
    det_match_logit_threshold: float = 0.3   # detect_and_match's det_threshold (a logit, not a probability)
    max_match_distance_um: float = 5.0
    grad_clip: float = 1.0
    augment: bool = True
    brightness_shift: float = 0.1
    eval_max_batches: int = 200               # their validation loop cap
    # ours
    seed: int = 0
    eval_tracking_every: int = 5
    save_every: int | None = 25
    select_by: str = "val_score"
    max_hours: float | None = None
    data_parallel: bool = True
    val_frac: float = VAL_FRAC
    val_seed: int = VAL_SEED
    val_prefix: str | None = None

    def model_config(self) -> dict:
        return {
            "unet_out_channels": self.unet_out_channels,
            "unet_layers": list(self.unet_layers),
            "downsample": list(DOWNSAMPLE),
            "window_size": self.window_size,
            "pool_kernel_um": self.pool_kernel_um,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unet_layers"] = list(self.unet_layers)
        return d


# --------------------------------------------------------------------------
# losses (verbatim)
# --------------------------------------------------------------------------

def compute_gt_transition_matrix(gt_ids_t: np.ndarray, gt_ids_t1: np.ndarray, edges: set[tuple[int, int]]) -> torch.Tensor:
    t_to_row = {int(n): i for i, n in enumerate(gt_ids_t)}
    t1_to_col = {int(n): i for i, n in enumerate(gt_ids_t1)}
    matrix = torch.zeros(len(gt_ids_t), len(gt_ids_t1), dtype=torch.float32)
    for s, t in edges:
        if s in t_to_row and t in t1_to_col:
            matrix[t_to_row[s], t1_to_col[t]] = 1.0
    return matrix


def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE on annotated rows and columns (sparse GT -- unannotated cells ignored)."""
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True, device=logits.device)
    probs = torch.softmax(logits, dim=0)  # dim=0 intentional: divisions allowed, merges aren't
    bce = F.binary_cross_entropy(probs, target, reduction="none")
    p_t = probs * target + (1 - probs) * (1 - target)
    loss = ((1 - p_t) ** 2) * bce
    return loss[mask].mean()


def compute_batch_loss(logits: torch.Tensor, target: torch.Tensor, mask_t: torch.Tensor, mask_t1: torch.Tensor) -> torch.Tensor:
    losses = []
    for b in range(logits.shape[0]):
        nt = int(mask_t[b].sum().item())
        nt1 = int(mask_t1[b].sum().item())
        losses.append(compute_loss(logits[b, :nt, :nt1], target[b, :nt, :nt1]))
    return torch.stack(losses).mean()


def _evaluate_pair(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, int, int]:
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    if not active_rows.any():
        return 0.0, 0, 0
    loss = compute_loss(logits, target).item()
    probs = torch.softmax(logits, dim=0)
    preds = (probs > 0.5).float()
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    correct = (preds[mask] == target[mask]).sum().item()
    total = mask.sum().item()
    return loss, int(correct), int(total)


def compute_detection_loss(det_logits: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor, neg_weight: float = 0.1) -> torch.Tensor:
    """BCE detection loss: GT node voxels positive, all others lightly penalised (count-normalised)."""
    B = det_logits.shape[0]
    spatial = det_logits.shape[2:]
    logits = det_logits[:, 0]
    target = torch.zeros_like(logits)
    nt = mask.sum(dim=1).long()
    for b in range(B):
        n_gt = int(nt[b])
        if n_gt <= 0:
            continue
        gt_coords = coords[b, :n_gt]
        zi = gt_coords[:, 0].long().clamp(0, spatial[0] - 1)
        yi = gt_coords[:, 1].long().clamp(0, spatial[1] - 1)
        xi = gt_coords[:, 2].long().clamp(0, spatial[2] - 1)
        target[b, zi, yi, xi] = 1.0
    n_pos = target.reshape(B, -1).sum(dim=1).clamp(min=1)
    n_neg = (target.numel() // B - n_pos).clamp(min=1)
    shape = (B,) + (1,) * len(spatial)
    w_pos = (1.0 / n_pos).reshape(shape)
    w_neg = (neg_weight / n_neg).reshape(shape)
    weight = torch.where(target == 1.0, w_pos, w_neg)
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum") / B


# --------------------------------------------------------------------------
# detect -> match -> edge targets (verbatim)
# --------------------------------------------------------------------------

def detect_and_match(
    det_logits: torch.Tensor,
    gt_coords: torch.Tensor,
    mask: torch.Tensor,
    image_shape: tuple[int, ...],
    det_threshold: float = 0.3,
    pool_kernel_um: float = 5.0,
    max_match_distance: float = 5.0,
    voxel_size: tuple[float, ...] | None = None,
    frame_index: int = 0,
    window_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    B = det_logits.shape[0]
    device = det_logits.device
    vs = torch.tensor(voxel_size, dtype=torch.float32, device=device) if voxel_size is not None else None
    if voxel_size is not None:
        pool_kernel = tuple(max(1, k if k % 2 == 1 else k + 1) for k in (max(1, round(pool_kernel_um / s)) for s in voxel_size))
    else:
        k = max(1, round(pool_kernel_um))
        pool_kernel = (k if k % 2 == 1 else k + 1,) * 3
    pad = tuple(k // 2 for k in pool_kernel)

    with torch.no_grad():
        pooled = F.max_pool3d(det_logits, pool_kernel, stride=1, padding=pad)
        is_peak = (det_logits == pooled) & (det_logits > det_threshold)
        peak_idx = torch.nonzero(is_peak[:, 0])
    batch_ids = peak_idx[:, 0]
    peak_coords = peak_idx[:, 1:].float()
    nt_per_sample = mask.sum(dim=1).long()

    sample_matches: list[torch.Tensor] = []
    sample_coords: list[torch.Tensor] = []
    max_det = 0
    for b in range(B):
        sel = batch_ids == b
        det_b = peak_coords[sel]
        n_det = det_b.shape[0]
        nt = int(nt_per_sample[b].item())
        gt_b = gt_coords[b, :nt]
        n_gt = gt_b.shape[0]
        matched = torch.full((n_det,), -1, dtype=torch.long, device=device)
        if n_det > 0 and n_gt > 0:
            dists = torch.cdist(det_b * vs, gt_b * vs) if vs is not None else torch.cdist(det_b, gt_b)
            min_d, min_i = dists.min(dim=1)
            order = min_d.argsort()
            gt_taken = torch.zeros(n_gt, dtype=torch.bool, device=device)
            for idx in order:
                if min_d[idx] > max_match_distance:
                    break
                gi = min_i[idx]
                if not gt_taken[gi]:
                    matched[idx] = gi
                    gt_taken[gi] = True
        sample_matches.append(matched)
        sample_coords.append(det_b)
        max_det = max(max_det, n_det)
    max_det = max(max_det, 1)

    padded_coords = torch.zeros(B, max_det, 3, device=device)
    padded_mask = torch.zeros(B, max_det, dtype=torch.bool, device=device)
    for b in range(B):
        n = sample_coords[b].shape[0]
        if n == 0:
            continue
        padded_coords[b, :n] = sample_coords[b]
        padded_mask[b, :n] = True
    t_col = torch.full((B, max_det, 1), frame_index, device=device, dtype=torch.float32)
    full_coords = torch.cat([t_col, padded_coords], dim=-1)
    pos_shape = (window_size,) + tuple(image_shape[1:]) if window_size is not None else tuple(image_shape)
    padded_pos = _pos_embed_torch(full_coords, pos_shape)
    return padded_coords, padded_pos, padded_mask, sample_matches


def build_matched_edge_targets(match_t, match_t1, gt_target: torch.Tensor, max_det_t: int, max_det_t1: int) -> torch.Tensor:
    B = gt_target.shape[0]
    target = torch.zeros(B, max_det_t, max_det_t1, device=gt_target.device)
    for b in range(B):
        mt, mt1 = match_t[b], match_t1[b]
        gt_trans = gt_target[b]
        n_t, n_t1 = mt.shape[0], mt1.shape[0]
        if n_t == 0 or n_t1 == 0:
            continue
        valid_mask = (mt >= 0).unsqueeze(1) & (mt1 >= 0).unsqueeze(0)
        block = gt_trans[mt.clamp(min=0)][:, mt1.clamp(min=0)] * valid_mask.float()
        target[b, :n_t, :n_t1] = block
    return target


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

@dataclass
class WindowMeta:
    volume: int                     # index into the dataset's volume list
    t_start: int
    coords: list[np.ndarray]        # W arrays (N_i, 3) grid-unit float32 GT coords
    ids: list[np.ndarray]           # W arrays of GT node ids
    targets: list[torch.Tensor]     # W-1 transition matrices


def brightness_augment(imgs: torch.Tensor, coords: torch.Tensor, masks: torch.Tensor, *, rng: np.random.Generator, shift_range: float = 0.1):
    shift = rng.uniform(-shift_range, shift_range)
    return imgs + shift, coords, masks


def flip_augment(imgs: torch.Tensor, coords: torch.Tensor, masks: torch.Tensor, *, rng: np.random.Generator):
    """Each of Z, Y, X flipped independently with p = 0.5; real coords mirrored, padding left at zero."""
    flip_mask = rng.random(3) < 0.5
    dims_to_flip = [1 + dim for dim, flip in enumerate(flip_mask) if flip]
    if not dims_to_flip:
        return imgs, coords, masks
    imgs = imgs.flip(dims=dims_to_flip)
    coords = coords.clone()
    shape = imgs.shape[1:]
    for dim in range(3):
        if flip_mask[dim]:
            dim_coords = coords[..., dim]
            dim_coords[masks] = shape[dim] - dim_coords[masks] - 1
            coords[..., dim] = dim_coords
    return imgs, coords, masks


class FrameWindowDataset(Dataset):
    """Full-frame ``W``-frame windows whose frames all carry GT, padded to ``max_nodes``."""

    def __init__(
        self,
        volumes: list[VolumeFrames],
        windows: list[WindowMeta],
        max_nodes: int,
        *,
        augment: bool,
        brightness_shift: float,
        seed: int,
    ) -> None:
        self.volumes = volumes
        self.windows = windows
        self.max_nodes = max_nodes
        self.augment = augment
        self.brightness_shift = brightness_shift
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.windows)

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            base = torch.initial_seed() if info is not None else self.seed
            self._rng = np.random.default_rng(base % (2**32))
        return self._rng

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        vol = self.volumes[w.volume]
        W = len(w.coords)
        M = self.max_nodes
        imgs = torch.from_numpy(vol.window(w.t_start, W))  # (W, Z, Y, X) float32
        coords = torch.zeros(W, M, 3, dtype=torch.float32)
        masks = torch.zeros(W, M, dtype=torch.bool)
        for i in range(W):
            n = len(w.coords[i])
            coords[i, :n] = torch.from_numpy(w.coords[i])
            masks[i, :n] = True
        targets = torch.zeros(W - 1, M, M, dtype=torch.float32)
        for i in range(W - 1):
            nt, nt1 = len(w.coords[i]), len(w.coords[i + 1])
            targets[i, :nt, :nt1] = w.targets[i]
        if self.augment:
            rng = self.rng()
            imgs, coords, masks = brightness_augment(imgs, coords, masks, rng=rng, shift_range=self.brightness_shift)
            imgs, coords, masks = flip_augment(imgs, coords, masks, rng=rng)
        return {
            "imgs": imgs.half(),
            "coords": coords,
            "masks": masks,
            "targets": targets,
            "image_shape": torch.tensor((vol.n_t, *vol.shape), dtype=torch.long),
            "voxel_size": torch.tensor([s * d for s, d in zip((1.625, 0.40625, 0.40625), DOWNSAMPLE)], dtype=torch.float32),
            "downsample": torch.tensor(DOWNSAMPLE, dtype=torch.float32),
        }


def load_volume_windows(train_dir: Path, cache_dir: Path | None, name: str, volume_index: int, window_size: int) -> tuple[VolumeFrames, list[WindowMeta]]:
    vol = VolumeFrames(train_dir / f"{name}.zarr", cache_path(cache_dir, name) if cache_dir else None)
    gt = read_geff(train_dir / f"{name}.geff")
    ds = np.array(DOWNSAMPLE, dtype=np.float32)
    by_t: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for t in np.unique(gt.t):
        sel = gt.t == t
        zyx = np.stack([gt.z[sel], gt.y[sel], gt.x[sel]], axis=1).astype(np.float32) / ds
        by_t[int(t)] = (zyx, gt.node_ids[sel])
    edges = {(int(a), int(b)) for a, b in gt.edges.tolist()}
    windows: list[WindowMeta] = []
    for t0 in range(vol.n_t - window_size + 1):
        frames = [by_t.get(t0 + i) for i in range(window_size)]
        if any(f is None or len(f[0]) == 0 for f in frames):
            continue
        coords = [f[0] for f in frames]
        ids = [f[1] for f in frames]
        targets = [compute_gt_transition_matrix(ids[i], ids[i + 1], edges) for i in range(window_size - 1)]
        windows.append(WindowMeta(volume_index, t0, coords, ids, targets))
    return vol, windows


# --------------------------------------------------------------------------
# epoch loops (verbatim logic)
# --------------------------------------------------------------------------

def _frame_features(model, unet_out, det_logits, coords, masks, image_shape, voxel_size, cfg: TrainConfig, W: int):
    frame_det = []
    for i in range(W):
        det_c, det_p, det_m, matches = detect_and_match(
            det_logits[i], coords[:, i], masks[:, i], image_shape,
            det_threshold=cfg.det_match_logit_threshold, pool_kernel_um=cfg.pool_kernel_um,
            max_match_distance=cfg.max_match_distance_um, voxel_size=voxel_size,
            frame_index=i, window_size=W,
        )
        unet_feat = model._index_features(unet_out[:, i], det_c, det_m)
        frame_det.append((det_c, det_p, det_m, matches, unet_feat))
    return frame_det


def train_epoch(model, loader, optimizer, device, cfg: TrainConfig, log_every: int = 200) -> dict:
    model.train()
    total_edge = total_det = 0.0
    n_samples = 0
    grad_norms = []
    t0 = time.perf_counter()
    for step, batch in enumerate(loader):
        imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shape = tuple(batch["image_shape"][0].tolist())
        voxel_size = tuple(batch["voxel_size"][0].tolist())
        ds_scale = batch["downsample"][0].to(device)
        B, W = imgs.shape[:2]

        unet_out, det_logits = model.encode(imgs)
        det_loss = sum(compute_detection_loss(det_logits[i], coords[:, i], masks[:, i], cfg.det_neg_weight) for i in range(W)) / W
        frame_det = _frame_features(model, unet_out, det_logits, coords, masks, image_shape, voxel_size, cfg, W)
        block_losses = []
        for i in range(W - 1):
            ns, nt = frame_det[i][0].shape[1], frame_det[i + 1][0].shape[1]
            pair_target = build_matched_edge_targets(frame_det[i][3], frame_det[i + 1][3], targets[:, i], ns, nt)
            edge_logits = model.predict_edges(
                frame_det[i][4], frame_det[i + 1][4],
                frame_det[i][0] * ds_scale, frame_det[i + 1][0] * ds_scale,
                frame_det[i][1], frame_det[i + 1][1], frame_det[i][2], frame_det[i + 1][2],
            )
            block_losses.append(compute_batch_loss(edge_logits, pair_target, frame_det[i][2], frame_det[i + 1][2]))
        edge_loss = sum(block_losses) / len(block_losses)
        loss = edge_loss + cfg.det_loss_weight * det_loss
        if not torch.isfinite(loss):
            raise TrainingDivergedError(f"non-finite loss at step {step}: edge={edge_loss.item()} det={det_loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        grad_norms.append(float(gn))
        total_edge += edge_loss.item() * B
        total_det += det_loss.item() * B
        n_samples += B
        if log_every and step % log_every == 0:
            print(f"    step {step}/{len(loader)} edge={edge_loss.item():.5f} det={det_loss.item():.5f} "
                  f"gn={float(gn):.2f} {time.perf_counter() - t0:.0f}s", flush=True)
    return {
        "edge_loss": total_edge / max(n_samples, 1),
        "det_loss": total_det / max(n_samples, 1),
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
        "steps": len(grad_norms),
    }


@torch.no_grad()
def evaluate_pack(model, loader, device, cfg: TrainConfig) -> dict:
    """The pack's validation: detect->match->predict; edge accuracy and GT node recall."""
    model.eval()
    total_loss, correct, total, n_pairs = 0.0, 0, 0, 0
    gt_matched, gt_total = 0, 0
    for bi, batch in enumerate(loader):
        if cfg.eval_max_batches and bi >= cfg.eval_max_batches:
            break
        imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shape = tuple(batch["image_shape"][0].tolist())
        voxel_size = tuple(batch["voxel_size"][0].tolist())
        ds_scale = batch["downsample"][0].to(device)
        B, W = imgs.shape[:2]
        unet_out, det_logits = model.encode(imgs)
        frame_det = _frame_features(model, unet_out, det_logits, coords, masks, image_shape, voxel_size, cfg, W)
        for i in range(W):
            for b in range(B):
                gt_total += int(masks[b, i].sum().item())
                gt_matched += int((frame_det[i][3][b] >= 0).sum().item())
        for i in range(W - 1):
            ns, nt = frame_det[i][0].shape[1], frame_det[i + 1][0].shape[1]
            pair_target = build_matched_edge_targets(frame_det[i][3], frame_det[i + 1][3], targets[:, i], ns, nt)
            pair_logits = model.predict_edges(
                frame_det[i][4], frame_det[i + 1][4],
                frame_det[i][0] * ds_scale, frame_det[i + 1][0] * ds_scale,
                frame_det[i][1], frame_det[i + 1][1], frame_det[i][2], frame_det[i + 1][2],
            )
            for b in range(B):
                ns_b = int(frame_det[i][2][b].sum().item())
                nt_b = int(frame_det[i + 1][2][b].sum().item())
                l, c, n = _evaluate_pair(pair_logits[b, :ns_b, :nt_b], pair_target[b, :ns_b, :nt_b])
                total_loss += l
                correct += c
                total += n
                n_pairs += 1
    acc = correct / max(total, 1)
    recall = gt_matched / max(gt_total, 1)
    return {"pack_val_loss": total_loss / max(n_pairs, 1), "pack_val_acc": acc, "pack_val_recall": recall, "pack_val_score": acc * recall}


@torch.no_grad()
def eval_tracking(model, device, train_dir: Path, val_names: list[str], window_size: int) -> dict:
    """Competition metric on the held-out volumes through the real inference path (stage ilp, no TTA)."""
    from cell_tracking.pack_predict import PredictConfig
    from cell_tracking.pipeline import Models, run_volume

    was_training = model.training
    model.eval()
    models = Models(primary=model, device=device, window_size=window_size)
    cfg = PredictConfig(det_tta=False, edge_feature_tta=False)
    results = []
    t0 = time.time()
    for name in val_names:
        res = run_volume(models, train_dir / f"{name}.zarr", predict_cfg=cfg, stage="ilp")
        gt = read_geff(train_dir / f"{name}.geff")
        results.append(score_volume_graphs(name, gt, track_graph_to_geff(res.graph, name)))
    s = summarize(results)
    out = {
        "val_score": s["score"], "val_adj": s["adj"], "val_det_recall": s["det_recall"],
        "val_node_ratio": s["node_ratio"], "val_tp": s["tp"], "val_fp": s["fp"], "val_fn": s["fn"],
        "val_score_seconds": time.time() - t0,
    }
    for emb, se in summarize_by_embryo(results).items():
        out[f"val_score_{emb}"] = se["score"]
    if was_training:
        model.train()
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _unwrap(model: UNetNodeTransformer) -> dict:
    return normalize_state_dict(model.state_dict())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    train_dir: Path,
    out_path: Path,
    cfg: TrainConfig,
    *,
    cache_dir: Path | None = None,
    names: list[str] | None = None,
    resume: bool = True,
    started_at: float | None = None,
    device: torch.device | None = None,
    log_every: int = 200,
) -> Path:
    train_dir, out_path = Path(train_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = started_at or time.time()
    _seed_everything(cfg.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_names = names or list_geff_datasets(train_dir)
    train_names, val_names = split_dataset_names(all_names, val_frac=cfg.val_frac, seed=cfg.val_seed, val_prefix=cfg.val_prefix)
    print(f"{len(train_names)} train / {len(val_names)} val volumes"
          + (f" (held-out embryo {cfg.val_prefix})" if cfg.val_prefix else ""), flush=True)

    def _load(split_names: list[str], label: str):
        vols, wins = [], []
        for i, n in enumerate(split_names):
            v, w = load_volume_windows(train_dir, cache_dir, n, i, cfg.window_size)
            vols.append(v)
            wins.extend(w)
        print(f"  {label}: {len(vols)} volumes, {len(wins)} windows, cached={all(v.cached for v in vols)}", flush=True)
        return vols, wins

    train_vols, train_windows = _load(train_names, "train")
    val_vols, val_windows = _load(val_names, "val")
    max_nodes = max(max(len(c) for c in w.coords) for w in train_windows + val_windows)
    print(f"max_nodes={max_nodes}", flush=True)

    train_ds = FrameWindowDataset(train_vols, train_windows, max_nodes, augment=cfg.augment, brightness_shift=cfg.brightness_shift, seed=cfg.seed)
    val_ds = FrameWindowDataset(val_vols, val_windows, max_nodes, augment=False, brightness_shift=0.0, seed=cfg.seed)
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    def worker_init_fn(worker_id: int) -> None:
        np.random.seed(torch.initial_seed() % 2**32)

    loader_kw = dict(num_workers=cfg.num_workers, prefetch_factor=2 if cfg.num_workers > 0 else None,
                     persistent_workers=cfg.num_workers > 0, pin_memory=False, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=g, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kw)

    model = build_model(cfg.model_config()).to(device)
    n_visible = torch.cuda.device_count() if device.type == "cuda" else 0
    if cfg.data_parallel and n_visible > 1:
        model.unet = nn.DataParallel(model.unet)
        print(f"DataParallel: UNet across {n_visible} GPUs", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,} epochs={cfg.epochs} lr={cfg.lr} batch={cfg.batch_size}", flush=True)

    history: list[dict] = []
    start_epoch = 0
    best_value: float | None = None
    best_epoch: int | None = None
    if resume and out_path.exists():
        ckpt = torch.load(out_path, map_location=device, weights_only=False)
        state = ckpt["state_dict"]
        if isinstance(model.unet, nn.DataParallel):
            state = {(k.replace("unet.", "unet.module.", 1) if k.startswith("unet.") else k): v for k, v in state.items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"])
        history = list(ckpt.get("history", []))
        for h in history:
            v = h.get(cfg.select_by)
            if v is not None and (best_value is None or v > best_value):
                best_value, best_epoch = v, h["epoch"]
        print(f"resumed from {out_path} at epoch {start_epoch} (best {cfg.select_by}={best_value} @ {best_epoch})", flush=True)
        if start_epoch >= cfg.epochs:
            print("already at target epochs; nothing to do", flush=True)
            return out_path

    def payload(epoch: int, extra: dict | None = None) -> dict:
        d = {
            "state_dict": _unwrap(model),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg.model_config(),
            "train_config": cfg.to_dict(),
            "history": history,
            "val_names": list(val_names),
            "val_prefix": cfg.val_prefix,
        }
        if extra:
            d.update(extra)
        return d

    def save(path: Path, epoch: int, extra: dict | None = None) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload(epoch, extra), tmp)
        tmp.replace(path)

    best_path = out_path.with_name(out_path.stem + "_best" + out_path.suffix)
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        stats = train_epoch(model, train_loader, optimizer, device, cfg, log_every=log_every)
        train_seconds = time.time() - t0
        t1 = time.time()
        stats.update(evaluate_pack(model, val_loader, device, cfg))
        stats["pack_val_seconds"] = time.time() - t1
        epoch_no = epoch + 1
        is_last = epoch_no == cfg.epochs
        if cfg.eval_tracking_every and (epoch_no % cfg.eval_tracking_every == 0 or is_last):
            stats.update(eval_tracking(model, device, train_dir, val_names, cfg.window_size))
        stats.update({"epoch": epoch_no, "train_seconds": train_seconds, "lr": cfg.lr})
        history.append(stats)

        selected = stats.get(cfg.select_by)
        is_best = selected is not None and (best_value is None or selected > best_value)
        if is_best:
            best_value, best_epoch = selected, epoch_no
            save(best_path, epoch_no, {"selected_by": cfg.select_by})
        save(out_path, epoch_no)
        if cfg.save_every and epoch_no % cfg.save_every == 0:
            save(out_path.with_name(f"{out_path.stem}_epoch{epoch_no}{out_path.suffix}"), epoch_no)
        hist_path = out_path.with_name(f"history_{out_path.stem}.json")
        hist_path.write_text(json.dumps(history, indent=1))

        msg = (f"epoch {epoch_no}/{cfg.epochs} edge={stats['edge_loss']:.5f} det={stats['det_loss']:.5f} "
               f"pack_val acc={stats['pack_val_acc']:.4f} recall={stats['pack_val_recall']:.4f} "
               f"gn_max={stats['grad_norm_max']:.1f} train={train_seconds:.0f}s")
        if "val_score" in stats:
            embs = " ".join(f"{k}={v:.4f}" for k, v in stats.items() if k.startswith("val_score_") and not k.endswith("seconds"))
            msg += (f" | val_score={stats['val_score']:.4f} ({embs}) det_recall={stats['val_det_recall']:.3f} "
                    f"node_ratio={stats['val_node_ratio']:.3f} [{stats['val_score_seconds']:.0f}s]")
        if is_best:
            msg += f"  * new best {cfg.select_by}"
        print(msg, flush=True)

        if cfg.max_hours is not None and (time.time() - started_at) / 3600.0 >= cfg.max_hours:
            print(f"max_hours={cfg.max_hours} reached after epoch {epoch_no}; stopping cleanly (resume to continue)", flush=True)
            break

    print(f"done: last epoch {history[-1]['epoch'] if history else start_epoch}, best {cfg.select_by}={best_value} @ epoch {best_epoch}", flush=True)
    return out_path

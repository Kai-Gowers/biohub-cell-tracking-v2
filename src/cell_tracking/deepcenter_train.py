"""Training for the DeepCenter center-heatmap detector (ported).

Port of ``train_full_frame_center_detector.py`` from the pilkwang
``biohub-deepcenter-unet3d-center-prior-v1`` artifact (``context/pack_deepcenter``,
retrieved 2026-09-11): full-resolution frame -> 4x4 XY block mean ->
per-frame percentile normalisation (50 / 99.5, clip -0.5 / 6.0) -> Gaussian
blob target (sigma 1 pooled voxel, max-combined) -> positive-unlabelled
weight map (w_pos 12 where heatmap > 0.05, w_bg 1 below the 40th intensity
percentile, w_ignore 0.05 elsewhere) -> weighted soft BCE normalised by the
weight sum -> AdamW lr 1e-3, weight decay 0, no scheduler, no clipping ->
best checkpoint by validation loss. Checkpoint dict keys match the shipped
``best.pt`` (``config``, ``model_state``, ``optimizer_state``, ``epoch``,
``best_score``, ``history``) so ``models.deepcenter.load_deepcenter`` reads it.

Kept faithful on purpose (recorded in the report rather than silently fixed):
the per-item flip RNG depends only on the item index, so each frame gets the
same flip every epoch; the pooled/original coordinate convention is
``y_pooled = y / 4`` at training time. Deviation from the artifact: the split
is this repo's held-out split (``split_dataset_names``) instead of the
artifact's embryo-prefix split, which put 128 volumes in validation and 71 in
training.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cell_tracking.config import VAL_FRAC, VAL_SEED
from cell_tracking.io_geff import list_geff_datasets, read_geff, split_dataset_names
from cell_tracking.io_zarr import read_array_meta, read_volume
from cell_tracking.models.deepcenter import DeepCenterUNet3D, block_mean_xy, normalize_dynamic_range


@dataclass
class DeepCenterTrainConfig:
    """The shipped ``config.json`` values."""

    seed: int = 2026
    pool_factor: int = 4
    base_channels: int = 24
    gauss_sigma: float = 1.0
    pos_thresh: float = 0.05
    bg_quantile: float = 0.40
    w_pos: float = 12.0
    w_bg: float = 1.0
    w_ignore: float = 0.05
    norm_lo_pct: float = 50.0
    norm_hi_pct: float = 99.5
    norm_clip_lo: float = -0.5
    norm_clip_hi: float = 6.0
    batch_size: int = 8
    epochs: int = 50
    frames_per_movie: int = 0
    movie_limit: int | None = None
    val_fraction: float = VAL_FRAC
    num_workers: int = 4
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None
    random_flip: bool = True
    brightness_jitter: float = 0.0
    # ours
    val_seed: int = VAL_SEED
    val_prefix: str | None = None
    val_batches: int | None = 24


def make_heatmap(pooled_shape: tuple[int, int, int], centers_zyx: np.ndarray, pool_factor: int, sigma: float) -> np.ndarray:
    heatmap = np.zeros(pooled_shape, dtype=np.float32)
    if centers_zyx.size == 0:
        return heatmap
    radius = max(1, int(math.ceil(3.0 * sigma)))
    sigma2 = float(sigma) ** 2
    z_max, y_max, x_max = pooled_shape
    for z0, y0, x0 in centers_zyx:
        center = np.array([z0, y0 / pool_factor, x0 / pool_factor], dtype=np.float32)
        zc, yc, xc = [float(v) for v in center]
        z_start = max(0, int(math.floor(zc)) - radius)
        z_stop = min(z_max, int(math.floor(zc)) + radius + 2)
        y_start = max(0, int(math.floor(yc)) - radius)
        y_stop = min(y_max, int(math.floor(yc)) + radius + 2)
        x_start = max(0, int(math.floor(xc)) - radius)
        x_stop = min(x_max, int(math.floor(xc)) + radius + 2)
        if z_start >= z_stop or y_start >= y_stop or x_start >= x_stop:
            continue
        zz = np.arange(z_start, z_stop, dtype=np.float32)[:, None, None]
        yy = np.arange(y_start, y_stop, dtype=np.float32)[None, :, None]
        xx = np.arange(x_start, x_stop, dtype=np.float32)[None, None, :]
        d2 = (zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2
        blob = np.exp(-0.5 * d2 / max(sigma2, 1e-6)).astype(np.float32)
        view = heatmap[z_start:z_stop, y_start:y_stop, x_start:x_stop]
        np.maximum(view, blob, out=view)
    return heatmap


def positive_unlabeled_weight_map(image: np.ndarray, heatmap: np.ndarray, cfg: DeepCenterTrainConfig) -> np.ndarray:
    weights = np.full(heatmap.shape, cfg.w_ignore, dtype=np.float32)
    bg_cutoff = float(np.quantile(image, cfg.bg_quantile))
    weights[image < bg_cutoff] = cfg.w_bg
    weights[heatmap > cfg.pos_thresh] = cfg.w_pos
    return weights


def flip_together(rng: np.random.Generator, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    shape = arrays[0].shape
    axes = tuple(axis for axis in range(len(shape)) if rng.random() < 0.5)
    if axes:
        arrays = tuple(np.flip(arr, axis=axes) for arr in arrays)
    return tuple(np.ascontiguousarray(arr, dtype=np.float32) for arr in arrays)


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted = loss * weights
    return weighted.sum() / torch.clamp(weights.sum(), min=1.0)


def load_sample(train_dir: Path, name: str) -> dict:
    zarr_path = train_dir / f"{name}.zarr"
    shape, dtype = read_array_meta(zarr_path)
    gt = read_geff(train_dir / f"{name}.geff")
    centers: dict[int, np.ndarray] = {}
    for t in np.unique(gt.t):
        sel = gt.t == t
        centers[int(t)] = np.stack([gt.z[sel], gt.y[sel], gt.x[sel]], axis=1).astype(np.float32)
    return {"name": name, "zarr": zarr_path, "shape": shape, "dtype": dtype, "centers_by_t": centers}


class FullFrameDataset(Dataset):
    def __init__(self, samples: list[dict], cfg: DeepCenterTrainConfig, training: bool) -> None:
        self.samples = samples
        self.cfg = cfg
        self.training = training
        self.items: list[tuple[int, int]] = []
        rng = np.random.default_rng(cfg.seed + (0 if training else 10_000))
        for sample_idx, sample in enumerate(samples):
            n_t = int(sample["shape"][0])
            if cfg.frames_per_movie and 0 < cfg.frames_per_movie < n_t:
                frames = sorted(rng.choice(n_t, size=cfg.frames_per_movie, replace=False).tolist())
            else:
                frames = list(range(n_t))
            self.items.extend((sample_idx, int(t)) for t in frames)
        if not self.items:
            raise ValueError("No training frames were selected.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        sample_idx, t = self.items[index]
        sample = self.samples[sample_idx]
        frame = read_volume(sample["zarr"], t, sample["shape"], sample["dtype"])
        pooled = block_mean_xy(frame, self.cfg.pool_factor)
        image = normalize_dynamic_range(pooled, self.cfg)
        target = make_heatmap(image.shape, sample["centers_by_t"].get(t, np.empty((0, 3), dtype=np.float32)),
                              self.cfg.pool_factor, self.cfg.gauss_sigma)
        weights = positive_unlabeled_weight_map(image, target, self.cfg)
        if self.training:
            rng = np.random.default_rng((self.cfg.seed + 1_000_003 * index) & 0xFFFFFFFF)
            if self.cfg.random_flip:
                image, target, weights = flip_together(rng, image, target, weights)
            if self.cfg.brightness_jitter > 0:
                scale = float(rng.uniform(1.0 - self.cfg.brightness_jitter, 1.0 + self.cfg.brightness_jitter))
                image = np.ascontiguousarray(image * scale, dtype=np.float32)
        return torch.from_numpy(image[None, ...]), torch.from_numpy(target[None, ...]), torch.from_numpy(weights[None, ...])


def save_checkpoint(path: Path, model, optimizer, cfg, epoch: int, best_score: float, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": asdict(cfg), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": int(epoch), "best_score": float(best_score), "history": history}, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None) -> float:
    model.eval()
    losses = []
    for batch_idx, (image, target, weights) in enumerate(loader, start=1):
        image = image.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        weights = weights.to(device=device, dtype=torch.float32)
        losses.append(float(weighted_bce_loss(model(image), target, weights).detach().cpu()))
        if max_batches is not None and batch_idx >= max_batches:
            break
    return float(np.mean(losses)) if losses else float("nan")


def train(
    train_dir: Path,
    output_dir: Path,
    cfg: DeepCenterTrainConfig,
    *,
    names: list[str] | None = None,
    resume: bool = True,
    progress_interval: int = 50,
    max_hours: float | None = None,
) -> Path:
    train_dir, output_dir = Path(train_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n")

    all_names = names or list_geff_datasets(train_dir)
    if cfg.movie_limit:
        all_names = all_names[: cfg.movie_limit]
    train_names, val_names = split_dataset_names(all_names, val_frac=cfg.val_fraction, seed=cfg.val_seed, val_prefix=cfg.val_prefix)
    samples_train = [load_sample(train_dir, n) for n in train_names]
    samples_val = [load_sample(train_dir, n) for n in val_names]
    (output_dir / "split_manifest.json").write_text(json.dumps(
        {"seed": cfg.val_seed, "val_fraction": cfg.val_fraction, "val_prefix": cfg.val_prefix,
         "train": train_names, "val": val_names}, indent=2) + "\n")
    print(f"samples: train={len(samples_train)} val={len(samples_val)}", flush=True)

    train_ds = FullFrameDataset(samples_train, cfg, training=True)
    val_ds = FullFrameDataset(samples_val or samples_train[:1], cfg, training=False)
    print(f"frames: train={len(train_ds)} val={len(val_ds)}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=torch.cuda.is_available(), drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=max(0, min(cfg.num_workers, 2)),
                            pin_memory=torch.cuda.is_available(), drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepCenterUNet3D(base_channels=cfg.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_path, last_path, history_path = output_dir / "best.pt", output_dir / "checkpoint_last.pt", output_dir / "history.csv"

    start_epoch, best_score, history = 0, float("-inf"), []
    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch, best_score, history = int(ckpt["epoch"]), float(ckpt["best_score"]), list(ckpt["history"])
        print(f"resumed from epoch {start_epoch}; best_score={best_score:.6f}", flush=True)

    total_batches = len(train_loader)
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        model.train()
        epoch_start = time.time()
        running, seen = 0.0, 0
        for batch_idx, (image, target, weights) in enumerate(train_loader, start=1):
            image = image.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            weights = weights.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_bce_loss(model(image), target, weights)
            loss.backward()
            if cfg.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            running += float(loss.detach().cpu())
            seen += 1
            if batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == total_batches:
                print(f"epoch={epoch}/{cfg.epochs} batch={batch_idx}/{total_batches} train_loss={running / max(seen, 1):.6f}", flush=True)
        train_loss = running / max(seen, 1)
        val_loss = evaluate(model, val_loader, device, cfg.val_batches)
        score = -val_loss if np.isfinite(val_loss) else -train_loss
        minutes = (time.time() - epoch_start) / 60.0
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss), "score": float(score), "minutes": float(minutes)})
        with open(history_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss", "score", "minutes"])
            w.writeheader()
            w.writerows(history)
        if score > best_score:
            best_score = score
            save_checkpoint(best_path, model, optimizer, cfg, epoch, best_score, history)
            print(f"new best epoch={epoch} val_loss={val_loss:.6f}", flush=True)
        save_checkpoint(last_path, model, optimizer, cfg, epoch, best_score, history)
        print(f"epoch={epoch} done train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_score={best_score:.6f} minutes={minutes:.2f}", flush=True)
        if max_hours is not None and (time.time() - started) / 3600.0 >= max_hours:
            print(f"max_hours={max_hours} reached; stopping cleanly", flush=True)
            break
    print(f"best: {best_path}  last: {last_path}", flush=True)
    return best_path

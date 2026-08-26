"""Train the detector: one head, one loss, one checkpoint.

Samples are single frames, not temporal windows -- there is no edge head to
feed, so nothing needs two frames from the same forward pass.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from cell_tracking.cache import VolumeFrames, cache_path
from cell_tracking.config import CHECKPOINT_NAME, VAL_FRAC, VAL_SEED, get_cache_dir, get_train_dir
from cell_tracking.io_geff import list_geff_datasets, read_geff, split_dataset_names
from cell_tracking.losses import detection_loss
from cell_tracking.models.detector import UNet3D


@dataclass
class Sample:
    frame: np.ndarray  # (Z, Y, X) float32
    node_grid: np.ndarray  # (n, 3) int grid coords of GT nodes in this frame


class VolumeSampler:
    """Frames from one volume, with ground-truth node positions attached."""

    def __init__(self, train_dir: Path, cache_dir: Path, name: str) -> None:
        from cell_tracking.config import voxels_to_grid

        self.name = name
        self.frames = VolumeFrames(train_dir / f"{name}.zarr", cache_path(cache_dir, name))
        graph = read_geff(train_dir / f"{name}.geff")

        coords: dict[int, list[tuple[float, float, float]]] = {}
        for i in range(len(graph.node_ids)):
            t = int(graph.t[i])
            if 0 <= t < self.frames.n_t:
                coords.setdefault(t, []).append(
                    (float(graph.z[i]), float(graph.y[i]), float(graph.x[i]))
                )
        self.grid = {
            t: np.rint(voxels_to_grid(np.asarray(v, dtype=np.float64))).astype(np.int64)
            for t, v in coords.items()
        }
        self.starts = list(range(self.frames.n_t))

    def __len__(self) -> int:
        return len(self.starts)

    def sample(self, t: int) -> Sample:
        return Sample(self.frames.frame(t), self.grid.get(t, np.zeros((0, 3), dtype=np.int64)))


def detection_target(shape_zyx: tuple[int, int, int], node_grid: np.ndarray) -> np.ndarray:
    """Binary target: 1 at each ground-truth node voxel, 0 everywhere else."""
    target = np.zeros(shape_zyx, dtype=np.float32)
    if len(node_grid) == 0:
        return target
    z = np.clip(node_grid[:, 0], 0, shape_zyx[0] - 1)
    y = np.clip(node_grid[:, 1], 0, shape_zyx[1] - 1)
    x = np.clip(node_grid[:, 2], 0, shape_zyx[2] - 1)
    target[z, y, x] = 1.0
    return target


def compute_loss(model: UNet3D, samples: list[Sample], device: torch.device) -> tuple[torch.Tensor, dict]:
    x = torch.from_numpy(np.stack([s.frame for s in samples])).unsqueeze(1).to(device)
    logits = model(x)[:, 0]
    targets = np.stack([detection_target(s.frame.shape, s.node_grid) for s in samples])
    y = torch.from_numpy(targets).to(device)
    loss = detection_loss(logits, y)
    n_gt = sum(len(s.node_grid) for s in samples) / len(samples)
    return loss, {"loss": float(loss.detach()), "n_gt": n_gt}


@dataclass
class History:
    entries: list[dict] = field(default_factory=list)

    def append(self, **kw) -> None:
        self.entries.append(kw)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.entries, indent=1))


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    train_dir: Path | None = None,
    out_path: Path | None = None,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    frames_per_volume: int = 20,
    val_frames_per_volume: int = 4,
    batch_size: int = 4,
    grad_accum: int = 2,
    cache_dir: Path | None = None,
    names: list[str] | None = None,
    val_frac: float = VAL_FRAC,
    val_seed: int = VAL_SEED,
    device: torch.device | None = None,
    amp: bool | None = None,
    resume: bool = True,
    seed: int = 0,
    history: History | None = None,
    log_every: int = 200,
    max_hours: float | None = None,
    started_at: float | None = None,
    select_by: str = "val_loss",
    patience: int | None = None,
) -> Path:
    """Train to `out_path`, checkpointing every epoch.

    `out_path` always holds the LAST epoch, so a run stays resumable. The
    best epoch by `select_by` is mirrored to a `_best` sibling -- inference
    should load that one; the previous project's separate repo lost 8.6
    GPU-hours to exactly this distinction going unenforced.

    `patience` stops after that many epochs with no `select_by` improvement.
    `max_hours` + `started_at` (a `time.time()` from the *start of the
    session*, before any cache build) stop cleanly after the last epoch that
    fits a Kaggle session, rather than losing a killed epoch's work.
    """
    train_dir = Path(train_dir or get_train_dir())
    cache_dir = Path(cache_dir or get_cache_dir())
    out_path = Path(out_path or (Path.cwd() / CHECKPOINT_NAME))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = device or _pick_device()
    amp = (device.type == "cuda") if amp is None else amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = history if history is not None else History()

    all_names = names or list_geff_datasets(train_dir)
    train_names, val_names = split_dataset_names(all_names, val_frac=val_frac, seed=val_seed)
    print(f"device={device}  volumes: {len(train_names)} train / {len(val_names)} val")

    model = UNet3D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    start_epoch = 0
    best_path = out_path.with_name(f"{out_path.stem}_best{out_path.suffix}")
    best_value = float("inf")
    best_epoch = 0

    if resume and out_path.exists():
        ckpt = torch.load(out_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
            if sched.T_max != epochs:
                print(f"rescheduling cosine T_max {sched.T_max} -> {epochs}")
                sched.T_max = epochs
        start_epoch = int(ckpt.get("epoch", 0))
        history.entries = list(ckpt.get("history", []))
        print(f"resumed from {out_path} at epoch {start_epoch}")
        best_value = min(
            (e[select_by] for e in history.entries if select_by in e), default=float("inf")
        )
        best_epoch = next(
            (e["epoch"] for e in history.entries if e.get(select_by) == best_value), 0
        )
        if start_epoch >= epochs:
            print("already trained for the requested number of epochs")
            return out_path

    print("loading volume index...")
    samplers = {n: VolumeSampler(train_dir, cache_dir, n) for n in train_names + val_names}
    cached = sum(1 for s in samplers.values() if s.frames.cached)
    print(f"{cached}/{len(samplers)} volumes cached (uncached volumes prepare frames on the fly)")

    def run_epoch(epoch: int, split: list[str], per_volume: int, training: bool) -> dict:
        model.train(training)
        rng = random.Random(seed * 1000 + epoch if training else 12345)
        work: list[tuple[str, int]] = []
        for n in split:
            s = samplers[n]
            if not len(s):
                continue
            starts = s.starts
            if training:
                picked = rng.sample(starts, min(per_volume, len(starts)))
            else:
                step = max(1, len(starts) // per_volume)
                picked = starts[::step][:per_volume]
            work.extend((n, t) for t in picked)
        rng.shuffle(work)

        totals: dict[str, float] = {}
        n_steps = 0
        t_start = time.time()
        n_batches = (len(work) + batch_size - 1) // batch_size
        every = min(log_every, max(1, n_batches // 10)) if log_every else 0
        if training:
            opt.zero_grad(set_to_none=True)
        for k in range(n_batches):
            chunk = work[k * batch_size : (k + 1) * batch_size]
            batch = [samplers[name].sample(t) for name, t in chunk]
            if training:
                with torch.autocast("cuda", enabled=amp):
                    loss, stats = compute_loss(model, batch, device)
                scaler.scale(loss / grad_accum).backward()
                if (k + 1) % grad_accum == 0 or k == n_batches - 1:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
            else:
                with torch.no_grad(), torch.autocast("cuda", enabled=amp):
                    _, stats = compute_loss(model, batch, device)
            for key, v in stats.items():
                totals[key] = totals.get(key, 0.0) + v
            n_steps += 1
            if training and every and n_steps % every == 0:
                rate = n_steps / (time.time() - t_start)
                print(
                    f"  epoch {epoch + 1} batch {n_steps}/{n_batches} "
                    f"loss={totals['loss'] / n_steps:.4f} ({rate:.2f} batch/s)"
                )
        return {k: v / max(n_steps, 1) for k, v in totals.items()} | {
            "steps": n_steps,
            "frames": len(work),
        }

    run_started = started_at if started_at is not None else time.time()
    if started_at is not None and max_hours is not None:
        setup = (time.time() - started_at) / 3600
        print(
            f"budget: {max_hours}h total, {setup:.2f}h already spent on setup "
            f"-> {max_hours - setup:.2f}h available for training"
        )
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr = run_epoch(epoch, train_names, frames_per_volume, True)
        va = run_epoch(epoch, val_names, val_frames_per_volume, False)
        sched.step()
        elapsed = time.time() - t0
        print(
            f"epoch {epoch + 1}/{epochs}  loss={tr['loss']:.4f}  val_loss={va['loss']:.4f}  "
            f"steps={tr['steps']}  {elapsed / 60:.1f} min"
        )
        history.append(
            epoch=epoch + 1,
            loss=tr["loss"],
            val_loss=va["loss"],
            lr=sched.get_last_lr()[0],
            seconds=elapsed,
        )
        payload = {
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch + 1,
            "config": {},
            "history": history.entries,
            "val_names": val_names,
        }
        torch.save(payload, out_path)

        current = history.entries[-1][select_by]
        if current < best_value:
            best_value, best_epoch = current, epoch + 1
            torch.save(payload | {"selected_by": select_by}, best_path)
            print(f"  new best {select_by}={best_value:.5f} -> {best_path.name}")
        elif patience is not None and (epoch + 1) - best_epoch >= patience:
            print(
                f"\nEarly stop at epoch {epoch + 1}/{epochs}: no {select_by} improvement "
                f"in {patience} epochs (best {best_value:.5f} at epoch {best_epoch})."
            )
            break

        if max_hours is not None:
            spent = (time.time() - run_started) / 3600
            if spent + (elapsed / 3600) > max_hours and epoch + 1 < epochs:
                print(
                    f"\nStopping at epoch {epoch + 1}/{epochs}: {spent:.1f}h spent, next "
                    f"epoch needs ~{elapsed / 60:.0f} min and the budget is {max_hours}h. "
                    "Checkpoint is resumable -- rerun with the same command to continue."
                )
                break

    print(f"Wrote {out_path} (last epoch)")
    if best_epoch:
        print(f"Wrote {best_path} (epoch {best_epoch}, {select_by}={best_value:.5f}) -- use this one")
    return out_path

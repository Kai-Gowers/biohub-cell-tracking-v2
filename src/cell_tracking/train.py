"""Train the detector and the edge scorer together, one checkpoint.

Detection samples are `WINDOW_SIZE`-frame windows (the temporal U-Net's
input), not single frames -- see `models/detector.py`. Edge-scorer samples
are detect-and-match frame pairs (`edge_train.py`): every `edge_every`-th
step, in the SAME backward pass as the detection loss, so the two heads train
jointly rather than the edge scorer bolting onto a frozen detector.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from cell_tracking.augment import augment_window
from cell_tracking.cache import VolumeFrames, cache_path
from cell_tracking.config import (
    CHECKPOINT_NAME,
    EDGE_LOSS_WEIGHT,
    VAL_FRAC,
    VAL_SEED,
    WINDOW_SIZE,
    get_cache_dir,
    get_train_dir,
    voxels_to_um,
)
from cell_tracking.edge_train import sample_edge_pair
from cell_tracking.io_geff import list_geff_datasets, read_geff, split_dataset_names
from cell_tracking.losses import detection_loss, edge_loss
from cell_tracking.models.detector import UNet3D
from cell_tracking.models.edge_model import EdgeScorer

# Tried splitting this into a tighter conv-only clip (CONV_GRAD_CLIP_NORM=0.5)
# to test whether cumulative conv/decoder growth was driving the recurring
# NaN divergence -- reverted after measuring a divergence at epoch 15,
# earlier than either directly-comparable same-LR joint-clip run (epoch 22,
# 23). No evidence it helped. See reports/2026-08-30-conv-grad-clip.md and
# reports/2026-08-30-nan-recurrence-after-conv-clip.md.


@dataclass
class Sample:
    window: np.ndarray  # (T, Z, Y, X) float32, target frame is the LAST one
    node_grid: np.ndarray  # (n, 3) int grid coords of GT nodes in the target frame


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

        # For edge-model detect-and-match: GT positions in µm (continuous,
        # not rounded to the model grid) plus their original node ids, and
        # the volume's real GT edge set (by node id).
        self.gt_um: dict[int, np.ndarray] = {}
        self.gt_ids: dict[int, np.ndarray] = {}
        for t, nodes in graph.nodes_by_t().items():
            if not (0 <= t < self.frames.n_t):
                continue
            ids = np.array([nid for nid, _ in nodes], dtype=np.int64)
            vox = np.array([c for _, c in nodes], dtype=np.float64)
            self.gt_ids[t] = ids
            self.gt_um[t] = voxels_to_um(vox)
        self.gt_edges = {(int(u), int(v)) for u, v in graph.edges}

        self.starts = list(range(self.frames.n_t))

    def __len__(self) -> int:
        return len(self.starts)

    def sample(
        self, t: int, *, augment: bool = False, rng: random.Random | None = None
    ) -> Sample:
        window = self.frames.window(t - (WINDOW_SIZE - 1), WINDOW_SIZE)
        grid = self.grid.get(t, np.zeros((0, 3), dtype=np.int64))
        if augment:
            window, grid = augment_window(window, grid, rng or random.Random())
        return Sample(window=window, node_grid=grid)


def _param_norm(*modules) -> float:
    """L2 norm of every parameter across `modules`, combined (not summed norms)."""
    total_sq = 0.0
    for m in modules:
        for p in m.parameters():
            total_sq += float(p.detach().float().pow(2).sum())
    return total_sq**0.5


def _grad_norm(param_list) -> float:
    """L2 norm of `.grad` across a parameter list.

    Also safe to call post-clip if ever needed elsewhere: if clipping
    actually triggered, every component was rescaled by the same factor, so
    the RATIO between components' norms survives even though the absolute
    values shrink. The per-group diagnostics below call this BEFORE
    `clip_grad_norm_`, though, specifically to get the true pre-clip
    magnitude of each group -- the post-clip ratio can't tell "barely over
    the cap" from "wildly over the cap," only the pre-clip value can."""
    total_sq = 0.0
    for p in param_list:
        if p.grad is not None:
            total_sq += float(p.grad.detach().float().pow(2).sum())
    return total_sq**0.5


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
    x = torch.from_numpy(np.stack([s.window for s in samples])).to(device)  # (B, T, Z, Y, X)
    logits = model(x)[:, 0]
    targets = np.stack([detection_target(s.window.shape[-3:], s.node_grid) for s in samples])
    y = torch.from_numpy(targets).to(device)
    loss = detection_loss(logits, y)
    n_gt = sum(len(s.node_grid) for s in samples) / len(samples)
    return loss, {"loss": float(loss.detach()), "n_gt": n_gt}


def compute_edge_loss(
    model: UNet3D,
    edge_scorer: EdgeScorer,
    samplers: dict[str, VolumeSampler],
    names: list[str],
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, dict, tuple[str, int]] | None:
    """Returns `(loss, stats, (volume_name, t))` -- the sampled volume/frame is
    reported alongside the loss so a non-finite event can be traced back to a
    specific source sample, not just "somewhere in training" (see
    TrainingDivergedError's docstring / run_epoch's `_batch_log`)."""
    candidates = [n for n in names if len(samplers[n]) >= 2]
    if not candidates:
        return None
    name = rng.choice(candidates)
    s = samplers[name]
    t = rng.randrange(0, s.frames.n_t - 1)
    out = sample_edge_pair(model, s, t, device)
    if out is None:
        return None
    feat_src, feat_dst, rel_um, y = out
    logits = edge_scorer(feat_src, feat_dst, rel_um)
    loss = edge_loss(logits, y)
    with torch.no_grad():
        pred = (torch.sigmoid(logits) >= 0.5).float()
        acc = float((pred == y).float().mean()) if len(y) else float("nan")
    return (
        loss,
        {
            "edge_loss": float(loss.detach()),
            "edge_pairs": len(y),
            "edge_pos": float(y.sum()),
            "edge_acc": acc,
        },
        (name, t),
    )


class TrainingDivergedError(RuntimeError):
    """Raised when the training loss goes non-finite (NaN/Inf) mid-epoch.

    Once a weight is actually NaN there is no self-recovery -- every later
    forward pass stays NaN too, `current < best_value` is always False for a
    NaN `current`, and `patience` only notices after grinding through that
    many more full epochs of pure wasted compute (observed: 13 epochs, ~50+
    minutes, before a patience=15 run caught it). Stopping immediately
    instead costs at most the rest of the epoch it happened in, and the last
    good `_best` checkpoint is untouched since it was written before this.
    """


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
    use_augment: bool = True,
    train_edge_model: bool = True,
    edge_loss_weight: float = EDGE_LOSS_WEIGHT,
    edge_every: int = 1,
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

    `train_edge_model` trains an `EdgeScorer` alongside the detector, one
    joint backward pass every `edge_every`-th step, via detect-and-match
    (`edge_train.py`) -- see that module's docstring for why its loss starts
    near zero and ramps up as detection quality improves.
    """
    train_dir = Path(train_dir or get_train_dir())
    cache_dir = Path(cache_dir or get_cache_dir())
    out_path = Path(out_path or (Path.cwd() / CHECKPOINT_NAME))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = device or _pick_device()
    if device.type == "cuda":
        # cuDNN benchmark mode: batch shapes here are essentially fixed, so
        # this is generally harmless/free, but measured NO effect on the
        # bf16 slowdown below (reports/2026-08-30-cudnn-benchmark-bf16-slowdown.md
        # first suspected it as the cause; scripts/bench_amp_dtype.py on a
        # real Kaggle T4 showed identical timings with this on or off --
        # ruled out, not a fix).
        torch.backends.cudnn.benchmark = True
    amp = (device.type == "cuda") if amp is None else amp
    # float16's ~65504 max and narrow dynamic range is a plausible driver of
    # the recurring NaN divergence chased across this project's history: the
    # conv/decoder stack's gradient sits at the clip ceiling essentially
    # every epoch from early in training (not an escalating rare event), and
    # GradScaler-caught inf/nan gradients recur every 6-8 epochs regardless
    # of edge-scorer variant -- see reports/2026-08-29-nan-divergence-bf16.md.
    # bfloat16 has float32's exponent range, so the same weight/activation
    # magnitudes can't overflow it, at the cost of less mantissa precision.
    #
    # torch.cuda.is_bf16_supported() is NOT a reliable guard for this: it
    # returned True on a Kaggle Tesla T4 (Turing, compute capability 7.5,
    # NO bf16 tensor cores at all), and bf16 measured ~7x slower than fp16
    # end-to-end there (Conv3d/InstanceNorm3d alone: ~20x slower; the
    # attention layers: only ~1.9x -- scripts/bench_amp_dtype.py,
    # reports/2026-08-30-cudnn-benchmark-bf16-slowdown.md). Check compute
    # capability directly instead: bf16 tensor-core acceleration needs
    # Ampere or newer (>= 8.0).
    if amp and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        autocast_dtype = torch.bfloat16
    elif amp and device.type == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32  # unused: torch.autocast(enabled=False) ignores dtype
    print(f"AMP: {'bf16' if autocast_dtype == torch.bfloat16 else ('fp16' if amp else 'disabled')}")
    # GradScaler's loss scaling exists specifically to counter float16
    # underflow; bfloat16 doesn't need it (matches float32's exponent
    # range), so only enable it when actually autocasting to float16.
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and autocast_dtype == torch.float16))
    history = history if history is not None else History()

    all_names = names or list_geff_datasets(train_dir)
    train_names, val_names = split_dataset_names(all_names, val_frac=val_frac, seed=val_seed)
    print(f"device={device}  volumes: {len(train_names)} train / {len(val_names)} val")

    model = UNet3D().to(device)
    edge_scorer = EdgeScorer().to(device) if train_edge_model else None
    params = list(model.parameters()) + (list(edge_scorer.parameters()) if edge_scorer else [])
    # Named subsets for per-component gradient-norm diagnostics (see
    # TrainingDivergedError's docstring / the diag print line below) -- kept
    # separate from `params` itself, which is what actually gets clipped and
    # optimized as one combined group.
    conv_params = (
        list(model.encoders.parameters())
        + list(model.decoders.parameters())
        + list(model.bottleneck.parameters())
        + list(model.out_proj.parameters())
    )
    attn_params = list(model.temporal_attn.parameters()) + list(model.bottleneck_attn.parameters())
    edge_params = list(edge_scorer.parameters()) if edge_scorer else []
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    start_epoch = 0
    best_path = out_path.with_name(f"{out_path.stem}_best{out_path.suffix}")
    best_value = float("inf")
    best_epoch = 0

    if resume and out_path.exists():
        ckpt = torch.load(out_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if edge_scorer is not None and ckpt.get("edge_state_dict") is not None:
            edge_scorer.load_state_dict(ckpt["edge_state_dict"])
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
        if edge_scorer is not None:
            edge_scorer.train(training)
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
        grad_norms: list[float] = []
        conv_grad_norms: list[float] = []
        attn_grad_norms: list[float] = []
        edge_grad_norms: list[float] = []
        n_skipped_steps = 0
        # Samples accumulated since the last optimizer step -- attached to a
        # caught inf/nan skip or a hard divergence so either can be traced
        # back to a specific source volume/frame, not just "somewhere this
        # epoch". Cleared after every step (skipped or not), since a skip is
        # specific to whatever was accumulated into that step's gradient.
        accum_samples: list[tuple[str, int]] = []
        if training:
            opt.zero_grad(set_to_none=True)
        for k in range(n_batches):
            chunk = work[k * batch_size : (k + 1) * batch_size]
            batch = [
                samplers[name].sample(t, augment=(training and use_augment), rng=rng)
                for name, t in chunk
            ]
            do_edge = edge_scorer is not None and edge_every > 0 and (n_steps % edge_every == 0)
            edge_sample: tuple[str, int] | None = None
            if training:
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=amp):
                    loss, stats = compute_loss(model, batch, device)
                    if do_edge:
                        edge_out = compute_edge_loss(model, edge_scorer, samplers, split, rng, device)
                        if edge_out is not None:
                            edge_l, edge_stats, edge_sample = edge_out
                            loss = loss + edge_loss_weight * edge_l
                            stats.update(edge_stats)
                accum_samples.extend(chunk)
                if edge_sample is not None:
                    accum_samples.append(edge_sample)
                if not torch.isfinite(loss):
                    exc = TrainingDivergedError(
                        f"non-finite loss ({float(loss.detach()):.4g}) at epoch {epoch + 1} "
                        f"batch {n_steps + 1}/{n_batches} -- stopping now rather than "
                        "continuing to train a corrupted model."
                    )
                    # This epoch's run_epoch() never returns, so its diagnostics
                    # would otherwise be lost -- exactly the data most worth
                    # having at the moment things actually broke.
                    exc.diag = {
                        "grad_norm_max": max(grad_norms) if grad_norms else float("nan"),
                        "grad_norm_mean": sum(grad_norms) / len(grad_norms) if grad_norms else float("nan"),
                        "conv_grad_norm_max": max(conv_grad_norms) if conv_grad_norms else float("nan"),
                        "attn_grad_norm_max": max(attn_grad_norms) if attn_grad_norms else float("nan"),
                        "edgescorer_grad_norm_max": max(edge_grad_norms) if edge_grad_norms else float("nan"),
                        "scaler_scale": scaler.get_scale(),
                        "n_skipped_steps": n_skipped_steps,
                        "batch_samples": list(accum_samples),
                    }
                    raise exc
                scaler.scale(loss / grad_accum).backward()
                if (k + 1) % grad_accum == 0 or k == n_batches - 1:
                    scaler.unscale_(opt)
                    # Split conv/other clipping (CONV_GRAD_CLIP_NORM=0.5) was
                    # tried and reverted: it measured a hard divergence at
                    # epoch 15, earlier than both directly-comparable
                    # same-LR joint-clip runs (epoch 22, 23) -- no evidence
                    # it helped, mild evidence it didn't. See
                    # reports/2026-08-30-conv-grad-clip.md for the full
                    # writeup and reports/2026-08-30-nan-recurrence-after-conv-clip.md
                    # for this result. Back to one joint clip.
                    #
                    # Per-group norms are measured HERE, before clipping, so
                    # they're the true pre-clip magnitude each group asked
                    # for -- measuring after `clip_grad_norm_` (as before)
                    # only shows each group's share of the already-shrunk
                    # joint vector, which can't distinguish "conv wanted 1.01"
                    # from "conv wanted 100": both look like ~1.0 post-clip
                    # whenever conv dominates the combined vector.
                    conv_grad_norms.append(_grad_norm(conv_params))
                    attn_grad_norms.append(_grad_norm(attn_params))
                    if edge_params:
                        edge_grad_norms.append(_grad_norm(edge_params))
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                    grad_norms.append(float(grad_norm))
                    # GradScaler backs off its scale the instant it finds an
                    # inf/nan gradient and silently skips that step's update --
                    # a leading indicator of instability well before a loss
                    # actually goes non-finite. Comparing scale before/after
                    # is the standard way to detect that skip (no public
                    # "did I skip" flag exists).
                    scale_before = scaler.get_scale()
                    scaler.step(opt)
                    scaler.update()
                    if scaler.get_scale() < scale_before:
                        n_skipped_steps += 1
                        print(f"    skipped step (caught inf/nan) samples: {accum_samples}")
                    opt.zero_grad(set_to_none=True)
                    accum_samples = []
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=amp):
                    _, stats = compute_loss(model, batch, device)
                    if do_edge:
                        edge_out = compute_edge_loss(model, edge_scorer, samplers, split, rng, device)
                        if edge_out is not None:
                            _, edge_stats, _ = edge_out
                            stats.update(edge_stats)
            for key, v in stats.items():
                totals[key] = totals.get(key, 0.0) + v
            n_steps += 1
            if training and every and n_steps % every == 0:
                rate = n_steps / (time.time() - t_start)
                print(
                    f"  epoch {epoch + 1} batch {n_steps}/{n_batches} "
                    f"loss={totals['loss'] / n_steps:.4f} ({rate:.2f} batch/s)"
                )
        diag: dict[str, float] = {}
        if training and grad_norms:
            diag = {
                "grad_norm_max": max(grad_norms),
                "grad_norm_mean": sum(grad_norms) / len(grad_norms),
                "conv_grad_norm_max": max(conv_grad_norms) if conv_grad_norms else float("nan"),
                "attn_grad_norm_max": max(attn_grad_norms) if attn_grad_norms else float("nan"),
                "edgescorer_grad_norm_max": max(edge_grad_norms) if edge_grad_norms else float("nan"),
                "scaler_scale": scaler.get_scale(),
                "n_skipped_steps": n_skipped_steps,
            }
        return {k: v / max(n_steps, 1) for k, v in totals.items()} | {
            "steps": n_steps,
            "frames": len(work),
        } | diag

    run_started = started_at if started_at is not None else time.time()
    if started_at is not None and max_hours is not None:
        setup = (time.time() - started_at) / 3600
        print(
            f"budget: {max_hours}h total, {setup:.2f}h already spent on setup "
            f"-> {max_hours - setup:.2f}h available for training"
        )
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        try:
            tr = run_epoch(epoch, train_names, frames_per_volume, True)
        except TrainingDivergedError as exc:
            print(f"\n{exc}")
            diag = getattr(exc, "diag", {})
            conv_norm = _param_norm(model.encoders, model.decoders, model.bottleneck, model.out_proj)
            attn_norm = _param_norm(model.temporal_attn, model.bottleneck_attn)
            edge_norm = _param_norm(edge_scorer) if edge_scorer is not None else 0.0
            print(
                f"  diag at divergence: grad_norm(max/mean)={diag.get('grad_norm_max', float('nan')):.3f}/"
                f"{diag.get('grad_norm_mean', float('nan')):.3f}  "
                f"grad_norm_max(conv/attn/edge)={diag.get('conv_grad_norm_max', float('nan')):.3f}/"
                f"{diag.get('attn_grad_norm_max', float('nan')):.3f}/"
                f"{diag.get('edgescorer_grad_norm_max', float('nan')):.3f}  "
                f"scaler_scale={diag.get('scaler_scale', float('nan')):.0f}  "
                f"skipped_steps_this_epoch={diag.get('n_skipped_steps', 0)}  "
                f"|W|_conv={conv_norm:.2f}  |W|_attn={attn_norm:.2f}  |W|_edge={edge_norm:.2f}"
            )
            print(f"  batch samples (volume, t) at divergence: {diag.get('batch_samples', [])}")
            if best_epoch:
                print(
                    f"Stopping: last good checkpoint is epoch {best_epoch} "
                    f"({select_by}={best_value:.5f}) at {best_path}."
                )
            else:
                print("Stopping: no epoch improved yet, so there is no _best checkpoint to fall back to.")
            break
        va = run_epoch(epoch, val_names, val_frames_per_volume, False)
        sched.step()
        elapsed = time.time() - t0
        edge_msg = ""
        if "edge_loss" in tr:
            edge_msg = f"  edge_loss={tr['edge_loss']:.4f} edge_acc={tr.get('edge_acc', float('nan')):.3f}"
        print(
            f"epoch {epoch + 1}/{epochs}  loss={tr['loss']:.4f}  val_loss={va['loss']:.4f}  "
            f"steps={tr['steps']}  {elapsed / 60:.1f} min{edge_msg}"
        )
        conv_norm = _param_norm(model.encoders, model.decoders, model.bottleneck, model.out_proj)
        attn_norm = _param_norm(model.temporal_attn, model.bottleneck_attn)
        edge_norm = _param_norm(edge_scorer) if edge_scorer is not None else 0.0
        print(
            f"  diag: grad_norm(max/mean)={tr.get('grad_norm_max', float('nan')):.3f}/"
            f"{tr.get('grad_norm_mean', float('nan')):.3f}  "
            f"grad_norm_max(conv/attn/edge)={tr.get('conv_grad_norm_max', float('nan')):.3f}/"
            f"{tr.get('attn_grad_norm_max', float('nan')):.3f}/"
            f"{tr.get('edgescorer_grad_norm_max', float('nan')):.3f}  "
            f"scaler_scale={tr.get('scaler_scale', float('nan')):.0f}  "
            f"skipped_steps={tr.get('n_skipped_steps', 0)}  "
            f"|W|_conv={conv_norm:.2f}  |W|_attn={attn_norm:.2f}  |W|_edge={edge_norm:.2f}"
        )
        history.append(
            epoch=epoch + 1,
            loss=tr["loss"],
            val_loss=va["loss"],
            lr=sched.get_last_lr()[0],
            seconds=elapsed,
            grad_norm_max=tr.get("grad_norm_max"),
            grad_norm_mean=tr.get("grad_norm_mean"),
            conv_grad_norm_max=tr.get("conv_grad_norm_max"),
            attn_grad_norm_max=tr.get("attn_grad_norm_max"),
            edgescorer_grad_norm_max=tr.get("edgescorer_grad_norm_max"),
            scaler_scale=tr.get("scaler_scale"),
            n_skipped_steps=tr.get("n_skipped_steps", 0),
            weight_norm_conv=conv_norm,
            weight_norm_attn=attn_norm,
            weight_norm_edge=edge_norm,
            **{k: v for k, v in tr.items() if k.startswith("edge_")},
            **{f"val_{k}": v for k, v in va.items() if k.startswith("edge_")},
        )
        payload = {
            "state_dict": model.state_dict(),
            "edge_state_dict": edge_scorer.state_dict() if edge_scorer is not None else None,
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

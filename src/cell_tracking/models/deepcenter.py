"""DeepCenter full-frame center-heatmap detector (ported).

Ported from the public pilkwang ``biohub-deepcenter-unet3d-center-prior-v1``
artifact (``context/pack_deepcenter``: ``train_full_frame_center_detector.py``)
and from cell 18 of ``context/biohub-0-942-lb-proxy-score-0-9417.ipynb``
(``_DCDeepCenterUNet3D``, ``_dc_pool_frame_xy``, ``_dc_normalize_dynamic_range``,
``deepcenter_heatmap_for_frame``, ``deepcenter_score_point``), retrieved
2026-09-11. The notebook uses it only as an *add-only gate*: a synthetic
gap-closing node is kept only if the heatmap around it clears a threshold.
It never removes detector output.

Coordinate contract (artifact manifest): the model runs on the XY mean-pooled
frame ``(Z, Y//4, X//4)``; a point ``(z, y, x)`` in original voxels is scored
at pooled ``(round(z), round(y/4), round(x/4))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DeepCenterUNet3D(nn.Module):
    """3-level 3D U-Net with isotropic pooling; input dims must be divisible by 8."""

    def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc1 = ConvBlock3d(in_channels, c)
        self.down1 = nn.MaxPool3d(2, 2)
        self.enc2 = ConvBlock3d(c, c * 2)
        self.down2 = nn.MaxPool3d(2, 2)
        self.enc3 = ConvBlock3d(c * 2, c * 4)
        self.down3 = nn.MaxPool3d(2, 2)
        self.bottleneck = ConvBlock3d(c * 4, c * 8)
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, 2, 2)
        self.dec3 = ConvBlock3d(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, 2)
        self.dec2 = ConvBlock3d(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, 2)
        self.dec1 = ConvBlock3d(c * 2, c)
        self.head = nn.Conv3d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def block_mean_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    """Non-overlapping ``factor x factor`` block mean in Y and X (Z untouched)."""
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2 = (y // factor) * factor
    x2 = (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def normalize_dynamic_range(volume: np.ndarray, cfg: object) -> np.ndarray:
    """Per-frame percentile normalisation (defaults 50 / 99.5 pct, clip -0.5 / 6.0)."""
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, float(getattr(cfg, "norm_lo_pct", 50.0))))
    hi = float(np.percentile(vol, float(getattr(cfg, "norm_hi_pct", 99.5))))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    ratio = (vol - lo) / (hi - lo)
    return np.clip(
        ratio,
        float(getattr(cfg, "norm_clip_lo", -0.5)),
        float(getattr(cfg, "norm_clip_hi", 6.0)),
    ).astype(np.float32)


@dataclass
class DeepCenter:
    """Loaded veto model plus its training config and a small heatmap cache."""

    model: DeepCenterUNet3D
    cfg: SimpleNamespace
    device: torch.device
    path: Path
    epoch: int
    score_win_z: int = 1
    score_win_yx: int = 2
    cache_max_frames: int = 8
    # 0.947-notebook option (BIOHUB_DEEPCENTER_TTA): average the logits over the D4 views of the
    # pooled frame before the sigmoid. Off by default = the 0.942 notebook's single view.
    tta: bool = False

    def heatmap(self, frame: np.ndarray, cache: dict, key) -> np.ndarray:
        """Sigmoid heatmap for one full-resolution frame (cached by ``key``)."""
        cached = cache.get(key)
        if cached is not None:
            return cached
        pool_factor = int(getattr(self.cfg, "pool_factor", 4))
        pooled = block_mean_xy(frame, pool_factor)
        image = normalize_dynamic_range(pooled, self.cfg)
        with torch.no_grad():
            tensor = torch.from_numpy(image[None, None, ...]).to(device=self.device, dtype=torch.float32)
            logits = self.model(tensor)
            if self.tta:
                acc, nv = logits.clone(), 1
                for dims in [(-1,), (-2,), (-2, -1)]:
                    acc = acc + self.model(tensor.flip(dims)).flip(dims)
                    nv += 1
                if tensor.shape[-1] == tensor.shape[-2]:
                    for k in (1, 3):
                        acc = acc + torch.rot90(self.model(torch.rot90(tensor, k, dims=(-2, -1))), -k, dims=(-2, -1))
                        nv += 1
                    acc = acc + self.model(tensor.transpose(-1, -2)).transpose(-1, -2)
                    nv += 1
                    at = torch.rot90(tensor, 1, dims=(-2, -1)).transpose(-1, -2)
                    acc = acc + torch.rot90(self.model(at).transpose(-1, -2), -1, dims=(-2, -1))
                    nv += 1
                logits = acc / nv
            hm = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        cache[key] = hm
        limit = max(1, int(self.cache_max_frames))
        while len(cache) > limit:
            cache.pop(next(iter(cache)))
        return hm

    def score_point(self, heatmap: np.ndarray, point: tuple[float, float, float]) -> float | None:
        """Max heatmap value in a ±win_z / ±win_yx pooled window around ``point`` (original voxels)."""
        if heatmap is None or heatmap.size == 0:
            return None
        pool_factor = int(getattr(self.cfg, "pool_factor", 4))
        z = int(round(float(point[0])))
        y = int(round(float(point[1]) / max(pool_factor, 1)))
        x = int(round(float(point[2]) / max(pool_factor, 1)))
        z0, z1 = max(0, z - self.score_win_z), min(heatmap.shape[0], z + self.score_win_z + 1)
        y0, y1 = max(0, y - self.score_win_yx), min(heatmap.shape[1], y + self.score_win_yx + 1)
        x0, x1 = max(0, x - self.score_win_yx), min(heatmap.shape[2], x + self.score_win_yx + 1)
        patch = heatmap[z0:z1, y0:y1, x0:x1]
        if patch.size == 0:
            return None
        score = float(np.max(patch))
        return score if np.isfinite(score) else None


def load_deepcenter(
    checkpoint_path: Path | str,
    device: torch.device | str | None = None,
    *,
    expected_epoch: int | None = 2,
) -> DeepCenter:
    """Load a DeepCenter checkpoint (dict with ``model_state``, ``config``, ``epoch``).

    ``expected_epoch`` mirrors the notebook's ``BIOHUB_DEEPCENTER_EXPECTED_EPOCH``
    guard (the shipped ``best.pt`` is epoch 2); pass ``None`` to skip the check.
    """
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"{checkpoint_path}: checkpoint has no model_state")
    epoch = int(checkpoint.get("epoch", -1))
    if expected_epoch is not None and expected_epoch > 0 and epoch != expected_epoch:
        raise ValueError(f"expected DeepCenter epoch {expected_epoch}, got {epoch}")
    cfg = SimpleNamespace(**checkpoint.get("config", {}))
    model = DeepCenterUNet3D(base_channels=int(getattr(cfg, "base_channels", 24)))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return DeepCenter(model=model, cfg=cfg, device=device, path=checkpoint_path, epoch=epoch)

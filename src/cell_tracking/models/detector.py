"""Detector: a single-frame 3D U-Net, one head, one objective.

    a(r) = q_theta(h_theta(V_t))(r),  p_t(r) = sigmoid(a(r))

This is deliberately the smallest model that can plausibly detect cell
centers. There is no temporal window and no edge head -- linking is a
distance-gated bipartite assignment (`link.py`), not learned. Add either back
only once a held-out score shows this baseline is missing something they
would fix, and record that evidence in `reports/`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_tracking.config import UNET_BASE_CHANNELS, UNET_DEPTH


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _match_spatial(t: torch.Tensor, target_zyx: tuple[int, int, int]) -> torch.Tensor:
    """Center-crop or pad spatial dims to target (Z,Y,X)."""
    _, _, z, y, x = t.shape
    tz, ty, tx = target_zyx
    if z > tz:
        s = (z - tz) // 2
        t = t[:, :, s : s + tz, :, :]
    if t.shape[3] > ty:
        s = (t.shape[3] - ty) // 2
        t = t[:, :, :, s : s + ty, :]
    if t.shape[4] > tx:
        s = (t.shape[4] - tx) // 2
        t = t[:, :, :, :, s : s + tx]
    pz, py, px = tz - t.shape[2], ty - t.shape[3], tx - t.shape[4]
    if pz > 0 or py > 0 or px > 0:
        t = F.pad(t, (px // 2, px - px // 2, py // 2, py - py // 2, pz // 2, pz - pz // 2))
    return t


class UNet3D(nn.Module):
    """(B, 1, Z, Y, X) -> (B, 1, Z, Y, X) per-voxel center logit."""

    def __init__(
        self,
        base_channels: int = UNET_BASE_CHANNELS,
        depth: int = UNET_DEPTH,
    ) -> None:
        super().__init__()
        chs = [base_channels * (2**i) for i in range(depth)]

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = 1
        for ch in chs:
            self.encoders.append(ConvBlock3d(prev, ch))
            self.pools.append(nn.MaxPool3d(2))
            prev = ch

        bottleneck_ch = chs[-1] * 2
        self.bottleneck = ConvBlock3d(chs[-1], bottleneck_ch)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = bottleneck_ch
        for ch in reversed(chs):
            self.upconvs.append(nn.ConvTranspose3d(prev, ch, 2, stride=2))
            self.decoders.append(ConvBlock3d(prev, ch))
            prev = ch
        self.out_proj = nn.Conv3d(chs[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Expected (B, 1, Z, Y, X), got {tuple(x.shape)}")

        h = x
        skips: list[torch.Tensor] = []
        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = up(h)
            h = _match_spatial(h, skip.shape[-3:])
            h = dec(torch.cat([h, skip], dim=1))
        return self.out_proj(h)

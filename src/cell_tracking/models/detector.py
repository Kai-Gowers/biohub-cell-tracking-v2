"""Detector: a 2-frame-window 3D U-Net with cross-frame attention, one head.

    a(r) = q_theta(h_theta(V_{t-1:t}))(r),  p_t(r) = sigmoid(a(r))

Each frame in the window is encoded independently (shared weights) through
every stage; at every stage EXCEPT full resolution, a multi-head
self-attention block mixes information across the window before pooling
further. The decoder then predicts a single frame -- the LAST one in the
window -- from the attention-enriched features, using that frame's own
attended stage outputs as skip connections.

This replaces the v2 baseline's single-frame, no-temporal-window detector,
per reports/2026-08-25-sample-solution-0.90-comparison.md item 4: the 0.90
sample solution attends at every encoder stage except full-res, richer than
this repo's original "none at all" and the v1 sibling repo's
bottleneck-only mixing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_tracking.config import ATTN_HEADS, UNET_BASE_CHANNELS, UNET_DEPTH, UNET_DROPOUT


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
        ]
        if dropout > 0:
            # Dropout3d zeroes whole channels, not independent voxels -- with
            # this much spatial correlation between neighbouring voxels,
            # element-wise dropout barely perturbs anything.
            layers.append(nn.Dropout3d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalAttention3d(nn.Module):
    """Self-attention across the window's T frames, independently per voxel.

    Input/output (B, T, C, Z, Y, X). Each spatial location is its own
    attention "batch" over a length-T token sequence -- cheap, since T is
    small (2), and the whole point is to let the model mix information
    between frames rather than within one frame's own receptive field.

    No dropout here, deliberately: `batch x Z x Y x X` gets flattened into
    `nn.MultiheadAttention`'s batch dimension below, which comfortably
    exceeds 65535 at this task's resolutions (e.g. `4 x 32^3` at the first
    attention stage). Nonzero dropout forces PyTorch's fused SDPA kernel
    down a path that tracks per-element dropout masks via an RNG limited to
    a 65535 batch size, and fails outright above it -- confirmed on Kaggle:
    "Efficient attention cannot produce valid seed and offset outputs when
    the batch size exceeds (65535)". Regularization here comes from
    `ConvBlock3d`'s `Dropout3d` and `EdgeScorer`'s dropout instead, neither
    of which has this limitation.
    """

    def __init__(self, channels: int, num_heads: int = ATTN_HEADS) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        b, t, c, z, y, x = feats.shape
        tokens = feats.permute(0, 3, 4, 5, 1, 2).reshape(-1, t, c)
        normed = self.norm(tokens)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attended
        return tokens.reshape(b, z, y, x, t, c).permute(0, 4, 5, 1, 2, 3)


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
    """(B, T, Z, Y, X) -> (B, 1, Z, Y, X) per-voxel center logit for the last frame."""

    def __init__(
        self,
        base_channels: int = UNET_BASE_CHANNELS,
        depth: int = UNET_DEPTH,
        attn_heads: int = ATTN_HEADS,
        dropout: float = UNET_DROPOUT,
    ) -> None:
        super().__init__()
        chs = [base_channels * (2**i) for i in range(depth)]

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        # No attention at stage 0 (full res) -- index i's attention lives at
        # self.temporal_attn[i - 1] for i > 0.
        self.temporal_attn = nn.ModuleList(
            TemporalAttention3d(ch, attn_heads) for ch in chs[1:]
        )
        prev = 1
        for i, ch in enumerate(chs):
            # Stage 0 is full resolution -- no dropout there, see UNET_DROPOUT's comment.
            self.encoders.append(ConvBlock3d(prev, ch, dropout=dropout if i > 0 else 0.0))
            self.pools.append(nn.MaxPool3d(2))
            prev = ch

        bottleneck_ch = chs[-1] * 2
        self.bottleneck = ConvBlock3d(chs[-1], bottleneck_ch, dropout=dropout)
        self.bottleneck_attn = TemporalAttention3d(bottleneck_ch, attn_heads)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = bottleneck_ch
        n_stages = len(chs)
        for i, ch in enumerate(reversed(chs)):
            self.upconvs.append(nn.ConvTranspose3d(prev, ch, 2, stride=2))
            # The last decoder stage is full resolution and feeds both
            # out_proj and the edge scorer's sampled features -- no dropout
            # there either, for the same reason stage 0 skips it.
            is_finest = i == n_stages - 1
            self.decoders.append(ConvBlock3d(prev, ch, dropout=0.0 if is_finest else dropout))
            prev = ch
        self.out_proj = nn.Conv3d(chs[0], 1, kernel_size=1)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected (B, T, Z, Y, X), got {tuple(x.shape)}")

        b, t = x.shape[0], x.shape[1]
        h = x.reshape(b * t, 1, *x.shape[2:])
        skips: list[torch.Tensor] = []
        for i, (enc, pool) in enumerate(zip(self.encoders, self.pools)):
            h = enc(h)
            c, z, y, xw = h.shape[1:]
            feats = h.reshape(b, t, c, z, y, xw)
            if i > 0:
                feats = self.temporal_attn[i - 1](feats)
            skips.append(feats[:, -1])  # target (last) frame only, for the decoder
            h = feats.reshape(b * t, c, z, y, xw)
            h = pool(h)

        h = self.bottleneck(h)
        c, z, y, xw = h.shape[1:]
        feats = self.bottleneck_attn(h.reshape(b, t, c, z, y, xw))
        h = feats[:, -1]

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = up(h)
            h = _match_spatial(h, skip.shape[-3:])
            h = dec(torch.cat([h, skip], dim=1))
        logits = self.out_proj(h)
        if return_features:
            return logits, h
        return logits

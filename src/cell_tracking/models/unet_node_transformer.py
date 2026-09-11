"""TemporalUNet3D encoder + SimpleNodeTransformer edge predictor (ported).

Ported from the public pilkwang ``biohub-tracking-support-pack`` (RoyerLab
Kaggle baseline), ``scripts/train_unet_transformer.py`` (the
``UNetNodeTransformer`` class, ``extract_pos_features``, ``_pos_embed_torch``,
``_POS_EMBED_DIM``) and ``scripts/predict_unet_transformer.py``
(``load_model`` / ``_DEFAULT_CONFIG``), retrieved 2026-09-11
(``context/pack_primary``). Model code is verbatim; ``load_pack_model`` adds
the ``unet.module.`` prefix normalisation the pack's trainer applies when it
saves DataParallel checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cell_tracking.models.node_transformer import SimpleNodeTransformer
from cell_tracking.models.temporal_unet import TemporalUNet3D

_POS_EMBED_DIM = 8   # per axis; total = 4 axes x _POS_EMBED_DIM = 32

# Architecture defaults; the shipped weights' config.json holds exactly these
# values (plus pool_kernel_um=5.0, which inference does NOT read -- the
# pack's PredictConfig uses 3.0 µm; both give a (3,3,3) kernel on this grid).
DEFAULT_MODEL_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "downsample": [1, 4, 4],
    "window_size": 2,
}


def extract_pos_features(
    coords: np.ndarray,
    image_shape: tuple[int, ...],
    pos_embed_dim: int = _POS_EMBED_DIM,
) -> np.ndarray:
    """Sinusoidal positional embeddings for node coordinates (no intensity term).

    Parameters
    ----------
    coords : np.ndarray
        (N, 4) with columns [t, z, y, x].
    image_shape : tuple
        Full image shape (T, Z, Y, X) used for normalisation.
    pos_embed_dim : int
        Half-dimension per axis (sin half + cos half).

    Returns
    -------
    np.ndarray
        Shape (N, 4 * pos_embed_dim), float32.
    """
    t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    norms = [c / max(s, 1) for c, s in zip([t, z, y, x], image_shape)]

    def _embed(vals: np.ndarray) -> np.ndarray:
        freqs = 2 ** np.arange(pos_embed_dim // 2)
        angles = vals[:, None] * freqs * np.pi
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([_embed(n) for n in norms], axis=1).astype(np.float32)


def _pos_embed_torch(
    coords: torch.Tensor,
    image_shape: tuple[int, ...],
    pos_embed_dim: int = _POS_EMBED_DIM,
) -> torch.Tensor:
    """Batched sinusoidal positional embeddings (pure torch, stays on device).

    coords : (*, 4) with columns [t, z, y, x]; image_shape : (T, Z, Y, X).
    Returns shape (*, 4 * pos_embed_dim).
    """
    shape_t = torch.tensor(image_shape, dtype=torch.float32, device=coords.device)
    norms = coords / shape_t.clamp(min=1)  # (*, 4)
    freqs = (2.0 ** torch.arange(pos_embed_dim // 2, device=coords.device, dtype=torch.float32)) * torch.pi
    parts = []
    for ax in range(4):
        angles = norms[..., ax].unsqueeze(-1) * freqs  # (*, D//2)
        parts.extend([angles.sin(), angles.cos()])
    return torch.cat(parts, dim=-1)


class UNetNodeTransformer(nn.Module):
    """TemporalUNet3D encoder + SimpleNodeTransformer edge predictor.

    Forward pass:
      1. Stack frames t and t+1 -> (B, 2, 1, *spatial) -> UNet -> (B, 2, C_feat, *spatial)
      2. Integer-index feature maps at node coords (round + clamp; differentiable)
      3. Concatenate with sinusoidal positional embeddings
      4. Cross-attention transformer -> (B, max_nodes, max_nodes) edge logits
    """

    def __init__(
        self,
        unet: nn.Module,
        unet_out_channels: int,
        pos_feat_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.unet = unet
        self.unet_out_channels = unet_out_channels

        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)

        self.transformer = SimpleNodeTransformer(
            feat_dim=unet_out_channels + pos_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout=dropout,
        )

    def _index_features(
        self,
        feat_maps: torch.Tensor,  # (B, C, *spatial)
        coords: torch.Tensor,     # (B, max_nodes, 3)
        mask: torch.Tensor,       # (B, max_nodes) bool
    ) -> torch.Tensor:
        """Integer-index feat_maps at node positions; padded slots -> zeros.

        Gradients flow through the *feature map values* but NOT through the
        coordinates (integer indexing is non-differentiable w.r.t. position).
        """
        B, C = feat_maps.shape[:2]
        spatial = feat_maps.shape[2:]
        max_nodes = coords.shape[1]

        out = torch.zeros(B, max_nodes, C, device=feat_maps.device, dtype=feat_maps.dtype)
        for b in range(B):
            nt = int(mask[b].sum().item())
            if nt == 0:
                continue
            z = coords[b, :nt, 0].long().clamp(0, spatial[0] - 1)
            y = coords[b, :nt, 1].long().clamp(0, spatial[1] - 1)
            x = coords[b, :nt, 2].long().clamp(0, spatial[2] - 1)
            out[b, :nt] = feat_maps[b, :, z, y, x].T
        return out

    def detect(
        self,
        frame: torch.Tensor,  # (*spatial) -- single pre-downsampled frame
    ) -> torch.Tensor:
        """Run UNet + detection head on a single frame (duplicated into a fake pair)."""
        pair = torch.stack([frame, frame], dim=0).unsqueeze(0).unsqueeze(2)  # (1, 2, 1, *spatial)
        unet_out = self.unet(pair)          # (1, 2, C_feat, *spatial)
        det = self.detect_head(unet_out[0, 0:1])  # (1, 1, *spatial)
        return det[0, 0]  # (*spatial)

    def encode(
        self,
        imgs: torch.Tensor,  # (B, W, *spatial) -- already downsampled
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run UNet encoder on W pre-downsampled frames.

        Returns ``(unet_out, det_logits)`` where *unet_out* is
        ``(B, W, C_feat, *spatial)`` and *det_logits* is a list of W
        tensors each ``(B, 1, *spatial)``.
        """
        window = imgs.unsqueeze(2)  # (B, W, 1, *spatial)
        unet_out = self.unet(window)  # (B, W, C_feat, *spatial)
        W = unet_out.shape[1]
        det_logits = [self.detect_head(unet_out[:, i]) for i in range(W)]
        return unet_out, det_logits

    def predict_edges(
        self,
        unet_feat_src: torch.Tensor,  # (B, N_src, C_feat) pre-indexed
        unet_feat_tgt: torch.Tensor,  # (B, N_tgt, C_feat) pre-indexed
        coords_src: torch.Tensor,     # (B, N_src, 3)
        coords_tgt: torch.Tensor,     # (B, N_tgt, 3)
        pos_feat_src: torch.Tensor,   # (B, N_src, pos_feat_dim)
        pos_feat_tgt: torch.Tensor,   # (B, N_tgt, pos_feat_dim)
        mask_src: torch.Tensor,       # (B, N_src) bool
        mask_tgt: torch.Tensor,       # (B, N_tgt) bool
    ) -> torch.Tensor:
        """Run transformer edge predictor on pre-indexed UNet features."""
        feat_src = torch.cat([unet_feat_src, pos_feat_src], dim=-1)
        feat_tgt = torch.cat([unet_feat_tgt, pos_feat_tgt], dim=-1)
        return self.transformer(feat_src, feat_tgt, coords_src, coords_tgt, mask_src, mask_tgt)


def build_model(config: dict | None = None) -> UNetNodeTransformer:
    """Construct the model from a pack-style config dict (defaults filled in)."""
    config = {**DEFAULT_MODEL_CONFIG, **(config or {})}
    if "downsample_factor" in config and "downsample" not in config:
        df = config["downsample_factor"]
        config["downsample"] = [df, df, df]
    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=config["unet_out_channels"],
        layers=config["unet_layers"],
    )
    return UNetNodeTransformer(
        unet=unet,
        unet_out_channels=config["unet_out_channels"],
        pos_feat_dim=4 * _POS_EMBED_DIM,
    )


def normalize_state_dict(state: dict) -> dict:
    """Strip a DataParallel ``unet.module.`` prefix so weights load on one GPU."""
    return {k.replace("unet.module.", "unet.", 1): v for k, v in state.items()}


def load_pack_model(
    weights_path: Path | str,
    device: torch.device | str,
    config_path: Path | str | None = None,
) -> tuple[UNetNodeTransformer, dict]:
    """Reconstruct ``UNetNodeTransformer`` from ``config.json`` + a state dict.

    ``weights_path`` may be a bare state dict (the pack's
    ``edge_predictor_best.pth``) or this repo's training checkpoint (a dict
    with ``state_dict`` and ``config`` keys). ``config.json`` is looked up
    next to the weights when ``config_path`` is None.

    Returns ``(model, config)`` with ``config`` containing at least
    ``window_size`` and ``downsample``.
    """
    weights_path = Path(weights_path)
    device = torch.device(device)
    payload = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        config = {**DEFAULT_MODEL_CONFIG, **payload.get("config", {})}
    else:
        state = payload
        cp = Path(config_path) if config_path else weights_path.parent / "config.json"
        if cp.exists():
            config = {**DEFAULT_MODEL_CONFIG, **json.loads(cp.read_text())}
        else:
            print(f"Warning: config.json not found at {cp}, using defaults.", flush=True)
            config = dict(DEFAULT_MODEL_CONFIG)
    config["downsample"] = tuple(config["downsample"])
    model = build_model(config)
    model.load_state_dict(normalize_state_dict(state))
    model.to(device)
    model.eval()
    return model, config

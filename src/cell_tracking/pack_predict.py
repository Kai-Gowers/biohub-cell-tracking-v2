"""Per-video detection + edge scoring, the 0.942 notebook's way (ported).

Port of ``predict_video`` from the pilkwang pack's
``scripts/predict_unet_transformer.py`` with the six runtime patches that cell
16 of ``context/biohub-0-942-lb-proxy-score-0-9417.ipynb`` applies on top
(retrieved 2026-09-11):

1. 8-view planar D4 detection TTA (identity, 3 flips, rot90 +/-, transpose,
   anti-transpose; z is never flipped), averaged in logit space.
2. Dual seed: a secondary model's detection logits are moment-matched to the
   primary's and blended (0.2 / 0.8); its edge logits are blended by
   ``low_margin_consensus``.
3. Retention guard: per frame, fall back to the primary logits if the blend
   keeps < 90 % of the primary's peaks.
4. Bidirectional harmonic fusion of the primary edge logits (w = 0.15).
5. Edge-feature TTA: the primary U-Net feature map is averaged over the same
   8 views before node features are gathered.
6. Secondary edge-feature TTA with weight 0.75.

Everything is expressed on the batched ``(1, n_src, n_tgt)`` logit tensor as
in the source, so ``dim=1`` there is the *source* axis (each target column
picks its parent); the 2-D ``raw`` matrix uses ``dim=0`` for the same thing.

Outputs are integer detection coordinates in original voxels (grid index * 4,
no half-cell offset -- faithful to the pack) plus ``(src, tgt, prob, dist)``
candidate edges, ``dist`` in grid units as in the pack (unused downstream).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from cell_tracking.io_zarr import read_array_meta, read_attrs, read_scale, read_volume
from cell_tracking.models.unet_node_transformer import UNetNodeTransformer, extract_pos_features
from cell_tracking.preprocess import DOWNSAMPLE, grid_shape, prepare_frame, video_quantiles


@dataclass
class PredictConfig:
    """Every inference knob of the 0.942 run, with its committed value as default."""

    # detection
    det_threshold: float = 0.965          # BIOHUB_DET_THRESHOLD
    det_tta: bool = True                  # TTA on/off
    det_tta_views: str = "d4"             # "d4" = notebook's 8 views; "flip" = pack's original 4 (y, x, yx flips)
    pool_kernel_um: float = 3.0           # pack PredictConfig default (config.json's 5.0 is not read)
    # edges
    edge_activation: str = "softmax"      # softmax over sources per target
    edge_threshold_single: float = 0.5    # pack default when no secondary model
    edge_threshold_dual: float = 0.48     # BIOHUB_DUAL_SEED_EDGE_THRESHOLD
    edge_feature_tta: bool = True         # BIOHUB_EDGE_FEATURE_TTA
    bidirectional_edge_weight: float = 0.15   # BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT (0 disables)
    bidirectional_fusion_mode: str = "harmonic_probability"
    # secondary (dual seed)
    secondary_detection_weight: float = 0.80  # BIOHUB_SECONDARY_DETECTION_WEIGHT
    secondary_edge_weight: float = 0.20       # BIOHUB_SECONDARY_EDGE_WEIGHT
    secondary_link_mode: str = "low_margin_consensus"
    secondary_low_margin_max: float = 0.35
    secondary_mix_temperature: float = 1.0
    min_candidate_retention: float = 0.90     # BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION
    secondary_edge_feature_tta: bool = True
    secondary_edge_feature_tta_weight: float = 0.75
    # greedy degree caps, used only when the ILP is off (pack __post_init__: 1 parent / 2 children)
    use_ilp: bool = True
    max_parents_per_node: int | None = None
    max_children_per_node: int | None = None

    def __post_init__(self) -> None:
        if not self.use_ilp:
            if self.max_parents_per_node is None:
                self.max_parents_per_node = 1
            if self.max_children_per_node is None:
                self.max_children_per_node = 2

    def edge_threshold(self, dual: bool) -> float:
        return self.edge_threshold_dual if dual else self.edge_threshold_single

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# helpers ported from the pack
# --------------------------------------------------------------------------

def pool_kernel_from_um(um: float, voxel_size: tuple[float, ...]) -> tuple[int, ...]:
    """Per-axis odd voxel kernel for a physical suppression distance."""
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def detect_cells_pooled(
    det_logits: torch.Tensor,
    t: int,
    det_threshold: float,
    pool_kernel: tuple[int, ...],
) -> np.ndarray:
    """Max-pool local maxima of (1, Z, Y, X) logits above ``sigmoid > det_threshold``.

    Returns (N, 4) int16 [t, z, y, x] in grid units.
    """
    logits = det_logits.unsqueeze(0)  # (1, 1, Z, Y, X)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])  # (N, 3)
    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)
    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)


# The eight planar views, in the notebook's accumulation order. Each entry is
# (forward transform, inverse transform) acting on the last two (y, x) dims.
_D4_VIEWS: list[tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]] = [
    (lambda v: v.flip((-1,)), lambda v: v.flip((-1,))),
    (lambda v: v.flip((-2,)), lambda v: v.flip((-2,))),
    (lambda v: v.flip((-2, -1)), lambda v: v.flip((-2, -1))),
    (lambda v: torch.rot90(v, 1, dims=(-2, -1)), lambda v: torch.rot90(v, -1, dims=(-2, -1))),
    (lambda v: torch.rot90(v, 3, dims=(-2, -1)), lambda v: torch.rot90(v, -3, dims=(-2, -1))),
    (lambda v: v.transpose(-1, -2), lambda v: v.transpose(-1, -2)),
    (
        lambda v: torch.rot90(v, 1, dims=(-2, -1)).transpose(-1, -2),
        lambda v: torch.rot90(v.transpose(-1, -2), -1, dims=(-2, -1)),
    ),
]


def encode_with_tta(
    model: UNetNodeTransformer,
    imgs: torch.Tensor,
    *,
    det_tta: bool,
    feature_tta: bool,
    views: str = "d4",
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None, int]:
    """``model.encode`` plus the notebook's 8-view TTA.

    Returns ``(unet_out_single, det_logits_avg, unet_out_tta_mean_or_None, n_views)``.
    Detection logits are averaged over the views; the feature map average is
    returned separately so the caller can apply the primary (replace) or
    secondary (0.25/0.75 blend) rule.
    """
    unet_out, det_logits = model.encode(imgs)
    W = len(det_logits)
    if not det_tta:
        return unet_out, det_logits, None, 1
    unet_acc = unet_out.clone() if feature_tta else None
    n_views = 1
    view_list = _D4_VIEWS if views == "d4" else _D4_VIEWS[:3]
    for fwd, inv in view_list:
        u_view, det_view = model.encode(fwd(imgs))
        for f in range(W):
            det_logits[f] = det_logits[f] + inv(det_view[f])
        if feature_tta:
            unet_acc = unet_acc + inv(u_view)
        del u_view, det_view
        n_views += 1
    for f in range(W):
        det_logits[f] = det_logits[f] / n_views
    feat_mean = unet_acc / n_views if feature_tta else None
    return unet_out, det_logits, feat_mean, n_views


def _moment_match(src: torch.Tensor, ref: torch.Tensor, dim: int | None) -> torch.Tensor:
    """Shift/scale ``src`` onto ``ref``'s mean/std (std ratio clamped to [0.5, 2])."""
    if dim is None:
        ref_center, src_center = ref.mean(), src.mean()
        ref_scale = ref.float().std(unbiased=False).clamp_min(1e-4)
        src_scale = src.float().std(unbiased=False).clamp_min(1e-4)
    else:
        ref_center = ref.mean(dim=dim, keepdim=True)
        src_center = src.mean(dim=dim, keepdim=True)
        ref_scale = ref.float().std(dim=dim, keepdim=True, unbiased=False).clamp_min(1e-4)
        src_scale = src.float().std(dim=dim, keepdim=True, unbiased=False).clamp_min(1e-4)
    ratio = (ref_scale / src_scale).clamp(0.5, 2.0).to(src.dtype)
    return (src - src_center) * ratio + ref_center


def bidirectional_harmonic_fusion(
    model: UNetNodeTransformer,
    edge_logits_pair: torch.Tensor,
    weight: float,
    feat_src, feat_tgt, coords_src, coords_tgt, pos_src, pos_tgt, mask_src, mask_tgt,
) -> torch.Tensor:
    """Notebook patch 4: fuse forward logits with a reverse-time pass (harmonic mean of probabilities)."""
    reverse_native = model.predict_edges(
        feat_tgt, feat_src, coords_tgt, coords_src, pos_tgt, pos_src, mask_tgt, mask_src,
    )  # (1, n_tgt, n_src)
    reverse_pair = reverse_native.transpose(1, 2)
    forward_center = edge_logits_pair.mean(dim=1, keepdim=True)
    forward_scale = edge_logits_pair.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    reverse_center = reverse_pair.mean(dim=1, keepdim=True)
    reverse_scale = reverse_pair.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    reverse_scale_ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0).to(reverse_pair.dtype)
    reverse_aligned = (reverse_pair - reverse_center) * reverse_scale_ratio + forward_center
    forward_prob = torch.softmax(edge_logits_pair.float(), dim=1).clamp_min(1e-8)
    reverse_prob = torch.softmax(reverse_aligned.float(), dim=1).clamp_min(1e-8)
    harmonic_prob = 1.0 / ((1.0 - weight) / forward_prob + weight / reverse_prob)
    harmonic_prob = harmonic_prob / harmonic_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
    harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))
    harmonic_center = harmonic_logits.mean(dim=1, keepdim=True)
    harmonic_scale = harmonic_logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    harmonic_scale_ratio = (forward_scale / harmonic_scale).clamp(0.5, 2.0)
    return ((harmonic_logits - harmonic_center) * harmonic_scale_ratio + forward_center).to(reverse_aligned.dtype)


def blend_secondary_edges(
    edge_logits_pair: torch.Tensor,
    secondary_logits_pair: torch.Tensor,
    n_src: int,
    cfg: PredictConfig,
) -> torch.Tensor:
    """Notebook patch 2 (edge half): mix the secondary model's edge logits into the primary's."""
    mode = cfg.secondary_link_mode
    w = cfg.secondary_edge_weight
    if mode == "raw":
        secondary_for_mix = secondary_logits_pair
        blend_weight: torch.Tensor | float = w
    elif mode in {"calibrated", "adaptive", "low_margin_consensus"}:
        secondary_for_mix = _moment_match(secondary_logits_pair, edge_logits_pair, dim=1)
        if mode == "calibrated":
            blend_weight = w
        elif mode == "adaptive":
            if n_src >= 2:
                primary_probs = torch.softmax(edge_logits_pair[0], dim=0)
                secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)
                p2 = torch.topk(primary_probs, k=2, dim=0)
                s2 = torch.topk(secondary_probs, k=2, dim=0)
                primary_margin = p2.values[0] - p2.values[1]
                secondary_margin = s2.values[0] - s2.values[1]
                local_weight = (w + secondary_margin - primary_margin).clamp(0.15, 0.75)
                same_parent = p2.indices[0].eq(s2.indices[0])
                local_weight = torch.where(
                    same_parent,
                    torch.maximum(local_weight, torch.full_like(local_weight, w)),
                    local_weight,
                )
                blend_weight = local_weight.view(1, 1, -1)
            else:
                blend_weight = w
        else:  # low_margin_consensus
            if n_src >= 2:
                primary_probs = torch.softmax(edge_logits_pair[0], dim=0)
                secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)
                p2 = torch.topk(primary_probs, k=2, dim=0)
                s2 = torch.topk(secondary_probs, k=2, dim=0)
                primary_margin = p2.values[0] - p2.values[1]
                same_parent = p2.indices[0].eq(s2.indices[0])
                uncertainty = ((cfg.secondary_low_margin_max - primary_margin) / cfg.secondary_low_margin_max).clamp(0.0, 1.0)
                local_weight = w * uncertainty
                local_weight = torch.where(same_parent, local_weight, torch.zeros_like(local_weight))
                blend_weight = local_weight.view(1, 1, -1)
            else:
                blend_weight = 0.0
    else:
        raise ValueError(f"Unsupported secondary link mode: {mode}")

    mixed = (1.0 - blend_weight) * edge_logits_pair + blend_weight * secondary_for_mix
    if cfg.secondary_mix_temperature != 1.0:
        center = mixed.mean(dim=1, keepdim=True)
        mixed = center + (mixed - center) / cfg.secondary_mix_temperature
    return mixed


# --------------------------------------------------------------------------
# frame access
# --------------------------------------------------------------------------

class PackVolume:
    """Model-grid frames for one video, normalised with the video's own quantiles."""

    def __init__(self, zarr_path: Path | str, cache_file: Path | str | None = None) -> None:
        self.zarr_path = Path(zarr_path)
        self.name = self.zarr_path.stem
        self.raw_shape, self.dtype = read_array_meta(self.zarr_path)
        self.n_t = int(self.raw_shape[0])
        self.q_low, self.q_high = video_quantiles(read_attrs(self.zarr_path))
        self.scale = read_scale(self.zarr_path)
        self.grid_shape = grid_shape(self.raw_shape[1:])
        self.voxel_size = tuple(s * d for s, d in zip(self.scale, DOWNSAMPLE))
        self._cached: np.ndarray | None = None
        if cache_file is not None and Path(cache_file).exists():
            self._cached = np.load(cache_file, mmap_mode="r")

    def raw_frame(self, t: int) -> np.ndarray:
        """Full-resolution (Z, Y, X) frame as stored (uint16)."""
        return read_volume(self.zarr_path, t, self.raw_shape, self.dtype)

    def frame(self, t: int) -> np.ndarray:
        """Normalised float32 model-grid frame."""
        if self._cached is not None:
            from cell_tracking.preprocess import pack_normalize

            return pack_normalize(np.asarray(self._cached[t]), self.q_low, self.q_high)
        return prepare_frame(self.raw_frame(t), self.q_low, self.q_high)


# --------------------------------------------------------------------------
# the per-video loop
# --------------------------------------------------------------------------

@dataclass
class PredictOutput:
    coords: np.ndarray                      # (N, 4) int64 [t, z, y, x], original voxels
    edges: list[tuple[int, int, float, float]]
    stats: dict = field(default_factory=dict)


@torch.no_grad()
def predict_video(
    model: UNetNodeTransformer,
    volume: PackVolume,
    device: torch.device,
    cfg: PredictConfig,
    *,
    secondary: UNetNodeTransformer | None = None,
    window_size: int = 2,
    max_frames: int | None = None,
    verbose: bool = False,
    dump_det_dir: Path | None = None,
) -> PredictOutput:
    """Sliding-window inference on one video (stride W-1 covers every consecutive pair once).

    ``dump_det_dir``: diagnostics only -- write the final per-frame detection probability map (after TTA,
    dual-seed blend and retention guard, i.e. exactly what the peak picker sees) as float16
    ``<dir>/<volume>/t<t:03d>.npy`` on the 1.625 um grid.
    """
    started = time.time()
    T = volume.n_t if max_frames is None else min(volume.n_t, max_frames)
    image_shape = (T,) + tuple(volume.grid_shape)
    ds_arr = np.array(DOWNSAMPLE, dtype=np.float32)
    ds_arr_t = torch.from_numpy(ds_arr).to(device)
    W = window_size
    pool_k = pool_kernel_from_um(cfg.pool_kernel_um, volume.voxel_size)
    dual = secondary is not None
    edge_threshold = cfg.edge_threshold(dual)
    bidir_w = float(cfg.bidirectional_edge_weight)
    stats: dict = {
        "frames": T,
        "pool_kernel": pool_k,
        "edge_threshold": edge_threshold,
        "dual_seed": dual,
        "retention_guard": [],
        "edge_feature_tta_delta": [],
    }

    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    coord_offset: dict[int, tuple[int, int]] = {}
    global_node_count = 0
    all_edges: list[tuple[int, int, float, float]] = []

    stride = max(W - 1, 1)
    window_starts = list(range(0, T - W + 1, stride))
    if not window_starts or window_starts[-1] + W < T:
        last = max(T - W, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    for ws in window_starts:
        frame_indices = list(range(ws, ws + W))
        imgs = torch.from_numpy(np.stack([volume.frame(t) for t in frame_indices])).unsqueeze(0).to(device)  # (1, W, Z, Y, X)

        # --- primary encode + TTA (patches 1, 5) ---
        unet_out, det_logits, feat_mean, n_views = encode_with_tta(
            model, imgs, det_tta=cfg.det_tta, feature_tta=cfg.det_tta and cfg.edge_feature_tta,
            views=cfg.det_tta_views,
        )
        if feat_mean is not None:
            stats["edge_feature_tta_delta"].append(float((feat_mean - unet_out).abs().mean()))
            unet_out = feat_mean

        # --- secondary encode, detection blend, retention guard (patches 2, 3, 6) ---
        secondary_unet_out = None
        if dual:
            s_unet_out, s_det_logits, s_feat_mean, _ = encode_with_tta(
                secondary, imgs,
                det_tta=cfg.det_tta and cfg.secondary_detection_weight > 0.0,
                feature_tta=cfg.det_tta and cfg.secondary_detection_weight > 0.0 and cfg.secondary_edge_feature_tta,
                views=cfg.det_tta_views,
            )
            if s_feat_mean is not None:
                w_tta = cfg.secondary_edge_feature_tta_weight
                s_unet_out = (1.0 - w_tta) * s_unet_out + w_tta * s_feat_mean
            secondary_unet_out = s_unet_out
            if cfg.secondary_detection_weight > 0.0:
                for f in range(W):
                    primary_det = det_logits[f]
                    secondary_aligned = _moment_match(s_det_logits[f], primary_det, dim=None)
                    blended_det = (1.0 - cfg.secondary_detection_weight) * primary_det + cfg.secondary_detection_weight * secondary_aligned
                    primary_candidates = len(detect_cells_pooled(primary_det[0], int(frame_indices[f]), cfg.det_threshold, pool_k))
                    blended_candidates = len(detect_cells_pooled(blended_det[0], int(frame_indices[f]), cfg.det_threshold, pool_k))
                    retention = blended_candidates / primary_candidates if primary_candidates else 1.0
                    use_primary = bool(primary_candidates > 0 and retention < cfg.min_candidate_retention)
                    det_logits[f] = primary_det if use_primary else blended_det
                    if int(frame_indices[f]) not in seen_frames:
                        stats["retention_guard"].append({
                            "frame": int(frame_indices[f]),
                            "primary_candidates": int(primary_candidates),
                            "blended_candidates": int(blended_candidates),
                            "retention": float(retention),
                            "use_primary": use_primary,
                        })
            del s_det_logits
        del imgs

        # --- detect cells in each frame (dedup across windows) ---
        for f_idx, t in enumerate(frame_indices):
            if t not in seen_frames:
                if dump_det_dir is not None:
                    d = Path(dump_det_dir) / volume.name
                    d.mkdir(parents=True, exist_ok=True)
                    np.save(d / f"t{t:03d}.npy", torch.sigmoid(det_logits[f_idx][0]).detach().cpu().numpy().astype(np.float16))
                arr = detect_cells_pooled(det_logits[f_idx][0], t, cfg.det_threshold, pool_k)
                coord_offset[t] = (global_node_count, global_node_count + len(arr))
                global_node_count += len(arr)
                coord_lists.append(arr)
                seen_frames.add(t)
        coords_so_far = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)

        # --- edge prediction for each consecutive pair in the window ---
        for f_idx in range(W - 1):
            t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]
            if (t_src, t_tgt) in seen_pairs:
                continue
            seen_pairs.add((t_src, t_tgt))
            if t_src not in coord_offset or t_tgt not in coord_offset:
                continue
            s_src, e_src = coord_offset[t_src]
            s_tgt, e_tgt = coord_offset[t_tgt]
            if e_src == s_src or e_tgt == s_tgt:
                continue
            c_src = coords_so_far[s_src:e_src]
            c_tgt = coords_so_far[s_tgt:e_tgt]
            n_src, n_tgt = len(c_src), len(c_tgt)
            idx_src = np.arange(s_src, e_src, dtype=np.int64)
            idx_tgt = np.arange(s_tgt, e_tgt, dtype=np.int64)

            p_coords_src = torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            p_coords_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            window_shape = (W,) + image_shape[1:]
            c_src_rel = c_src.copy(); c_src_rel[:, 0] = f_idx
            c_tgt_rel = c_tgt.copy(); c_tgt_rel[:, 0] = f_idx + 1
            p_pos_src = torch.from_numpy(extract_pos_features(c_src_rel, window_shape)).unsqueeze(0).to(device)
            p_pos_tgt = torch.from_numpy(extract_pos_features(c_tgt_rel, window_shape)).unsqueeze(0).to(device)
            p_mask_src = torch.ones(1, n_src, dtype=torch.bool, device=device)
            p_mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=device)

            unet_feat_src = model._index_features(unet_out[:, f_idx], p_coords_src, p_mask_src)
            unet_feat_tgt = model._index_features(unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt)
            args = (p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t, p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt)
            edge_logits_pair = model.predict_edges(unet_feat_src, unet_feat_tgt, *args)  # (1, n_src, n_tgt)

            if bidir_w > 0.0:
                edge_logits_pair = bidirectional_harmonic_fusion(
                    model, edge_logits_pair, bidir_w, unet_feat_src, unet_feat_tgt, *args,
                )
            if dual:
                s_feat_src = secondary._index_features(secondary_unet_out[:, f_idx], p_coords_src, p_mask_src)
                s_feat_tgt = secondary._index_features(secondary_unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt)
                secondary_logits_pair = secondary.predict_edges(s_feat_src, s_feat_tgt, *args)
                edge_logits_pair = blend_secondary_edges(edge_logits_pair, secondary_logits_pair, n_src, cfg)

            raw = edge_logits_pair[0]
            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()

            ii, jj = np.nonzero(probs > edge_threshold)
            candidates = sorted(zip(probs[ii, jj].tolist(), ii.tolist(), jj.tolist()), reverse=True)
            children_count: dict[int, int] = {}
            parents_count: dict[int, int] = {}
            for prob, i, j in candidates:
                n_ch = children_count.get(i, 0)
                n_pa = parents_count.get(j, 0)
                if cfg.max_children_per_node is not None and n_ch >= cfg.max_children_per_node:
                    continue
                if cfg.max_parents_per_node is not None and n_pa >= cfg.max_parents_per_node:
                    continue
                gi, gj = int(idx_src[i]), int(idx_tgt[j])
                dist = float(np.linalg.norm(
                    coords_so_far[gi, 1:].astype(np.float32) - coords_so_far[gj, 1:].astype(np.float32)
                ))
                all_edges.append((gi, gj, float(prob), dist))
                children_count[i] = n_ch + 1
                parents_count[j] = n_pa + 1
        del unet_out, secondary_unet_out
        if verbose and ws % 20 == 0:
            print(f"    window {ws}/{T}: {global_node_count} nodes, {len(all_edges)} edges", flush=True)

    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr
    coords = coords.astype(np.int16).astype(np.int64)
    stats["detections"] = int(len(coords))
    stats["candidate_edges"] = len(all_edges)
    stats["retention_fallback_frames"] = sum(1 for r in stats["retention_guard"] if r["use_primary"])
    stats["predict_seconds"] = time.time() - started
    return PredictOutput(coords=coords, edges=all_edges, stats=stats)

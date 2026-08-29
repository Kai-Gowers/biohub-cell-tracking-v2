"""Per-volume inference: detect every frame, then link consecutive frames.

No repair stage in this baseline: linking only ever emits dt=1 edges with
in/out-degree <= 1, so `TrackGraph.validate()` holds by construction.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from cell_tracking.cache import VolumeFrames, cache_path
from cell_tracking.config import (
    LINK_RADIUS_UM,
    LINK_SCORE_THRESHOLD,
    MAX_DETECTIONS_PER_FRAME,
    TAU,
    WINDOW_SIZE,
    get_cache_dir,
    voxels_to_grid,
)
from cell_tracking.detect import FrameDetections, extract_detections, predict_logits_tta
from cell_tracking.graph import TrackGraph
from cell_tracking.io_geff import write_geff
from cell_tracking.link import link_frames
from cell_tracking.models.detector import UNet3D
from cell_tracking.models.edge_model import EdgeScorer, sample_node_features
from cell_tracking.peaks import pair_within_radius


def predict_volume(
    model: UNet3D,
    device: torch.device,
    zarr_path: Path | str,
    *,
    cache_dir: Path | None = None,
    tau: float = TAU,
    max_per_frame: int = MAX_DETECTIONS_PER_FRAME,
    subvoxel: bool = True,
    t_max: int | None = None,
    edge_scorer: EdgeScorer | None = None,
    tta: bool = True,
    link_radius_um: float = LINK_RADIUS_UM,
    link_score_threshold: float = LINK_SCORE_THRESHOLD,
) -> tuple[TrackGraph, dict]:
    """Track one volume end to end: per-frame detection, then frame-pair linking.

    `edge_scorer`, if given, scores every distance-gated candidate pair and
    `link.link_frames` uses those probabilities (thresholded at
    `link_score_threshold`) instead of falling back to negative distance.
    `tta` averages detection logits over identity + 3 flips (see
    `detect.predict_logits_tta`); it has no effect on edge scoring.
    """
    zarr_path = Path(zarr_path)
    name = zarr_path.stem
    cache_dir = Path(cache_dir) if cache_dir is not None else get_cache_dir()
    frames = VolumeFrames(zarr_path, cache_path(cache_dir, name))
    n_t = frames.n_t if t_max is None else min(frames.n_t, t_max)

    t_start = time.time()
    detections: dict[int, FrameDetections] = {}
    node_feats: dict[int, torch.Tensor] = {}
    for t in range(n_t):
        window = frames.window(t - (WINDOW_SIZE - 1), WINDOW_SIZE)
        logits, feats = predict_logits_tta(model, window, device, tta=tta)
        prob = torch.sigmoid(logits)
        det = extract_detections(
            prob, t, logits=logits, tau=tau, max_per_frame=max_per_frame, subvoxel=subvoxel
        )
        detections[t] = det
        if edge_scorer is not None:
            if len(det):
                grid_f = torch.from_numpy(voxels_to_grid(det.zyx)).float().to(device)
                node_feats[t] = sample_node_features(feats, grid_f)
            else:
                node_feats[t] = feats.new_zeros((0, feats.shape[0]))
    infer_time = time.time() - t_start

    graph = TrackGraph()
    node_ids: dict[int, np.ndarray] = {}
    for t in range(n_t):
        det = detections[t]
        node_ids[t] = np.array(
            [graph.add_node(t, det.zyx[k], float(det.score[k])) for k in range(len(det))],
            dtype=np.int64,
        )

    for t in range(n_t - 1):
        src, dst = detections[t], detections[t + 1]
        pairs = None
        edge_scores = None
        if edge_scorer is not None and len(src) and len(dst):
            pairs = pair_within_radius(src.um, dst.um, link_radius_um)
            if len(pairs):
                fs = node_feats[t][pairs[:, 0]]
                fd = node_feats[t + 1][pairs[:, 1]]
                rel_um = torch.from_numpy(dst.um[pairs[:, 1]] - src.um[pairs[:, 0]]).float().to(device)
                with torch.no_grad():
                    edge_logits = edge_scorer(fs, fd, rel_um)
                edge_scores = torch.sigmoid(edge_logits).detach().cpu().numpy()
        selected = link_frames(
            src.um,
            dst.um,
            radius_um=link_radius_um,
            pairs=pairs,
            edge_scores=edge_scores,
            score_threshold=link_score_threshold,
        )
        for a, b in selected:
            graph.add_edge(int(node_ids[t][a]), int(node_ids[t + 1][b]))

    graph.validate()

    stats = {
        "name": name,
        "frames": n_t,
        "detections": sum(len(d) for d in detections.values()),
        "detections_per_frame": sum(len(d) for d in detections.values()) / max(n_t, 1),
        "max_prob_mean": float(np.mean([d.max_prob for d in detections.values()])),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "infer_seconds": infer_time,
    }
    return graph, stats


def write_prediction(graph: TrackGraph, out_dir: Path | str, name: str) -> Path:
    """Write one graph as `<out_dir>/<name>.geff`."""
    arrays = graph.to_arrays()
    return write_geff(Path(out_dir) / f"{name}.geff", **arrays)

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
from cell_tracking.config import MAX_DETECTIONS_PER_FRAME, TAU, get_cache_dir
from cell_tracking.detect import FrameDetections, extract_detections
from cell_tracking.graph import TrackGraph
from cell_tracking.io_geff import write_geff
from cell_tracking.link import link_frames
from cell_tracking.models.detector import UNet3D


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
) -> tuple[TrackGraph, dict]:
    """Track one volume end to end: per-frame detection, then frame-pair linking."""
    zarr_path = Path(zarr_path)
    name = zarr_path.stem
    cache_dir = Path(cache_dir) if cache_dir is not None else get_cache_dir()
    frames = VolumeFrames(zarr_path, cache_path(cache_dir, name))
    n_t = frames.n_t if t_max is None else min(frames.n_t, t_max)

    t_start = time.time()
    detections: dict[int, FrameDetections] = {}
    for t in range(n_t):
        x = torch.from_numpy(frames.frame(t)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0, 0]
            prob = torch.sigmoid(logits)
        detections[t] = extract_detections(
            prob, t, logits=logits, tau=tau, max_per_frame=max_per_frame, subvoxel=subvoxel
        )
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
        selected = link_frames(src.um, dst.um)
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

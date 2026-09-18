"""Per-volume orchestrator for the 0.942 replication.

    predict (pack_predict) -> ILP (ilp) -> post-processing (postprocess)
    -> TrackGraph -> .geff

Everything stays in memory between stages so ``edge_prob`` reaches the
motion relink; ``stage`` stops early for parity checks and ablations:

* ``"raw"``  -- detections + thresholded candidate edges (no ILP)
* ``"ilp"``  -- after the tracksdata ILP (what the pack's ``.geff`` holds)
* ``"full"`` -- after the notebook's post-processing (what the CSV holds)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from cell_tracking.graph import TrackGraph
from cell_tracking.ilp import ILPConfig, graph_to_arrays, solve_ilp
from cell_tracking.io_geff import GeffGraph, write_geff
from cell_tracking.models.deepcenter import DeepCenter, load_deepcenter
from cell_tracking.models.unet_node_transformer import UNetNodeTransformer, load_pack_model
from cell_tracking.pack_predict import PackVolume, PredictConfig, predict_video
from cell_tracking.postprocess import PostprocessConfig, filter_output_graph

STAGES = ("raw", "ilp", "full")


@dataclass
class Models:
    primary: UNetNodeTransformer
    device: torch.device
    window_size: int = 2
    secondary: UNetNodeTransformer | None = None
    deepcenter: DeepCenter | None = None


def load_models(
    checkpoint: Path | str,
    secondary: Path | str | None = None,
    deepcenter: Path | str | None = None,
    device: torch.device | str | None = None,
    *,
    deepcenter_expected_epoch: int | None = 2,
) -> Models:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    primary, cfg = load_pack_model(checkpoint, device)
    window_size = int(cfg["window_size"])
    sec = None
    if secondary is not None:
        sec, scfg = load_pack_model(secondary, device)
        if int(scfg["window_size"]) != window_size or tuple(scfg["downsample"]) != tuple(cfg["downsample"]):
            raise ValueError("Primary and secondary models have incompatible inference grids")
    dc = load_deepcenter(deepcenter, device, expected_epoch=deepcenter_expected_epoch) if deepcenter is not None else None
    return Models(primary=primary, device=device, window_size=window_size, secondary=sec, deepcenter=dc)


def raw_to_nodes_edges(coords: np.ndarray, edges: list[tuple[int, int, float, float]]) -> tuple[dict[int, dict], list[dict]]:
    """Candidate graph without the ILP, in the post-processing's in-memory format (ids = row index)."""
    nodes_by_id = {
        int(i): {"node_id": int(i), "t": int(c[0]), "z": float(c[1]), "y": float(c[2]), "x": float(c[3])}
        for i, c in enumerate(coords)
    }
    out = [{"source_id": int(s), "target_id": int(t), "edge_prob": float(p)} for s, t, p, _ in edges]
    return nodes_by_id, out


def nodes_edges_to_graph(nodes_by_id: dict[int, dict], edges: list[dict]) -> tuple[TrackGraph, np.ndarray | None]:
    """In-memory nodes/edges -> TrackGraph (ids preserved) + aligned edge_prob array (NaN if unknown)."""
    ids = sorted(nodes_by_id)
    t = [nodes_by_id[i]["t"] for i in ids]
    zyx = [(nodes_by_id[i]["z"], nodes_by_id[i]["y"], nodes_by_id[i]["x"]) for i in ids]
    pairs = sorted(((int(e["source_id"]), int(e["target_id"])), e.get("edge_prob")) for e in edges)
    edge_arr = np.array([p for p, _ in pairs], dtype=np.int64).reshape(-1, 2)
    probs = np.array([np.nan if p is None else float(p) for _, p in pairs], dtype=np.float64)
    graph = TrackGraph.from_arrays(ids, t, zyx, edge_arr)
    return graph, probs


@dataclass
class VolumeResult:
    graph: TrackGraph
    edge_prob: np.ndarray | None
    stats: dict = field(default_factory=dict)


def run_volume(
    models: Models,
    zarr_path: Path | str,
    *,
    predict_cfg: PredictConfig = PredictConfig(),
    ilp_cfg: ILPConfig = ILPConfig(),
    post_cfg: PostprocessConfig = PostprocessConfig(),
    stage: str = "full",
    use_ilp: bool = True,
    max_frames: int | None = None,
    verbose: bool = False,
    dump_det_dir: Path | None = None,
) -> VolumeResult:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    started = time.time()
    volume = PackVolume(zarr_path)
    stats: dict = {"name": volume.name}

    out = predict_video(
        models.primary, volume, models.device, predict_cfg,
        secondary=models.secondary, window_size=models.window_size,
        max_frames=max_frames, verbose=verbose, dump_det_dir=dump_det_dir,
    )
    stats["predict"] = {k: v for k, v in out.stats.items() if k not in ("retention_guard", "edge_feature_tta_delta")}
    stats["predict"]["edge_feature_tta_delta_mean"] = (
        float(np.mean(out.stats["edge_feature_tta_delta"])) if out.stats["edge_feature_tta_delta"] else None
    )
    stats["retention_guard"] = out.stats["retention_guard"]

    if stage == "raw" or not use_ilp:
        nodes_by_id, edges = raw_to_nodes_edges(out.coords, out.edges)
        stats["ilp"] = {"ilp_solved": 0}
    else:
        nodes_by_id, edges, ilp_stats = solve_ilp(out.coords, out.edges, ilp_cfg, use_ilp=True)
        stats["ilp"] = ilp_stats

    if stage == "full":
        nodes_by_id, edges, post_stats = filter_output_graph(
            nodes_by_id, edges,
            cfg=post_cfg,
            frame_source=volume.raw_frame,
            deepcenter=models.deepcenter,
            name=volume.name,
            verbose=verbose,
        )
        stats["postprocess"] = post_stats

    graph, edge_prob = nodes_edges_to_graph(nodes_by_id, edges)
    graph.validate()
    stats["nodes"] = len(graph.nodes)
    stats["edges"] = len(graph.edges)
    stats["divisions"] = graph.n_divisions()
    stats["seconds"] = time.time() - started
    return VolumeResult(graph=graph, edge_prob=edge_prob, stats=stats)


def write_result(result: VolumeResult, out_dir: Path | str, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = result.graph.to_arrays()
    return write_geff(out_dir / f"{name}.geff", **arrays, edge_prob=result.edge_prob)


def geff_to_nodes_edges(g: GeffGraph) -> tuple[dict[int, dict], list[dict]]:
    """A `.geff` (e.g. an ILP-stage dump or the pack's output) -> post-processing inputs."""
    nodes_by_id = {
        int(g.node_ids[i]): {"node_id": int(g.node_ids[i]), "t": int(g.t[i]),
                              "z": float(g.z[i]), "y": float(g.y[i]), "x": float(g.x[i])}
        for i in range(len(g.node_ids))
    }
    edges = []
    for k, (u, v) in enumerate(g.edges.tolist()):
        p = None
        if g.edge_prob is not None and np.isfinite(g.edge_prob[k]):
            p = float(g.edge_prob[k])
        edges.append({"source_id": int(u), "target_id": int(v), "edge_prob": p})
    return nodes_by_id, edges

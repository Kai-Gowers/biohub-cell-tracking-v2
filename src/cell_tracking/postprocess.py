"""Graph post-processing of the 0.942 notebook (ported).

Port of cell 18 of ``context/biohub-0-942-lb-proxy-score-0-9417.ipynb``
(retrieved 2026-09-11): ``filter_output_graph`` and its passes, with the
~60 ``BIOHUB_*`` environment variables replaced by :class:`PostprocessConfig`
whose defaults are the committed v29/v30 values (cell 8 + cell 12). Decision
logic is unchanged; only plumbing differs:

* image intensities come from a ``frame_source(t) -> (Z, Y, X)`` callable
  (full-resolution voxels) instead of a hard-bound ``TEST_DIR``;
* the DeepCenter veto is a :class:`cell_tracking.models.deepcenter.DeepCenter`
  instead of a module-global bundle;
* stats keys the notebook declared but never incremented
  (``safe_division_geometric_candidates`` etc.) are dropped.

Order of operations (``filter_output_graph``): edge sanity filter -> motion
relink (replaces every ILP edge) -> single-parent repair -> gap-1 closing
(density-adaptive, synthetic midpoint refined on intensities, DeepCenter
add-only gate) -> strict gap-2 recovery -> safe divisions -> [division
geometry filter, off] -> prune isolated -> short-track filter + adaptive
rescue -> line-fit smoothing.

Nodes are ``{node_id: {"node_id", "t", "z", "y", "x", ...}}`` in original
voxels; edges are dicts with ``source_id``, ``target_id``, ``edge_prob``
(``None`` for repaired edges) and, after the sanity filter, ``distance_um``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from cell_tracking.models.deepcenter import DeepCenter

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)

FrameSource = Callable[[int], np.ndarray]


@dataclass
class PostprocessConfig:
    """Cell-8 / cell-12 values of the committed 0.942 run."""

    # edge sanity
    output_edge_max_um: float = 14.0
    output_enforce_next_frame: bool = True
    output_single_parent_repair: bool = True
    output_single_child_repair: bool = False
    output_prune_isolated: bool = True
    # motion relink
    output_motion_relink: bool = True
    motion_relink_tight_um: float = 5.5
    motion_relink_relaxed_um: float = 10.0
    motion_relink_velocity_weight: float = 0.5
    motion_relink_learned_bonus: float = 1.0
    motion_relink_max_frame_nodes: int = 2600
    # division geometry filter (off in the 0.942 run)
    output_division_geometry_filter: bool = False
    div_parent_max_um: float = 10.5
    div_sister_max_um: float = 8.0
    div_drop_to_single_if_bad: bool = True
    # gap-1 closing
    output_gap_close: bool = True
    gap_close_max_gap: int = 2            # clamped to 1 by the notebook (effective_gap_max)
    gap_close_um: float = 5.8
    gap_density_adaptive: bool = True
    gap_density_reference_um: float = 6.5
    gap_density_gain: float = 0.040
    gap_density_max_step_delta_um: float = 0.125
    gap_density_neighbors: int = 3
    gap_close_reuse_existing: bool = True
    gap_close_reuse_um: float = 3.2
    gap_close_max_added_frac: float = 0.05
    gap_close_max_added_abs: int = 2000
    gap_refine_synthetic: bool = True
    gap_refine_win_z: int = 1
    gap_refine_win_yx: int = 3
    gap_refine_max_shift_um: float = 3.2
    # short tracks
    output_filter_short_tracks: bool = True
    output_min_track_len: int = 6
    output_keep_division_components: bool = True
    adaptive_short_track_rescue: bool = True
    short_track_rescue_trigger_removed_frac: float = 0.10
    short_track_rescue_min_len: int = 4
    short_track_rescue_min_mean_edge_prob: float = 0.88
    short_track_rescue_max_mean_edge_dist_um: float = 3.0
    short_track_rescue_max_nodes_frac: float = 0.012
    short_track_rescue_max_nodes_abs: int = 120
    # line fit
    output_linefit_smooth: bool = True
    output_linefit_weight: float = 0.8
    output_linefit_window: int = 2
    # gap-2
    output_gap2_recovery: bool = True
    gap2_max_total_um: float = 10.2
    gap2_max_step_um: float = 4.4
    gap2_max_links_frac: float = 0.0045
    gap2_max_links_abs: int = 180
    gap2_require_context: bool = True
    gap2_frame_frac_cap: float = 0.006
    # safe divisions
    output_safe_divisions: bool = True
    safe_div_max_um: float = 9.0
    safe_div_sister_max_um: float = 14.0
    safe_div_sister_symmetry_tau: float = 0.6
    safe_div_existing_child_max_um: float = 10.0
    safe_div_frame_frac_cap: float = 0.0076
    safe_div_global_frac_cap: float = 0.00375
    safe_div_diverge_um: float = 2.25
    # DeepCenter add-only gate
    use_deepcenter_veto: bool = True
    deepcenter_gap_veto: bool = True
    deepcenter_safe_div_veto: bool = False
    deepcenter_gap_threshold: float = 0.25
    deepcenter_gap_confirm_min_span_um: float = 8.5
    deepcenter_safe_div_threshold: float = 0.26
    deepcenter_score_win_z: int = 1
    deepcenter_score_win_yx: int = 2
    deepcenter_score_cache_max_frames: int = 8
    deepcenter_tta: bool = False          # 0.947 notebook: D4-averaged DeepCenter logits (BIOHUB_DEEPCENTER_TTA)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def edge_distance_um(source: dict, target: dict) -> float:
    dz = (float(source["z"]) - float(target["z"])) * VOXEL_SCALE_UM[0]
    dy = (float(source["y"]) - float(target["y"])) * VOXEL_SCALE_UM[1]
    dx = (float(source["x"]) - float(target["x"])) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def point_distance_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dz = (a[0] - b[0]) * VOXEL_SCALE_UM[0]
    dy = (a[1] - b[1]) * VOXEL_SCALE_UM[1]
    dx = (a[2] - b[2]) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def node_point(node: dict) -> tuple[float, float, float]:
    return (float(node["z"]), float(node["y"]), float(node["x"]))


def edge_sort_key(edge: dict) -> tuple[float, float]:
    prob = edge.get("edge_prob")
    prob_value = float(prob) if prob is not None else 0.0
    return prob_value, -float(edge["distance_um"])


def _next_node_id(nodes_by_id: dict[int, dict]) -> int:
    return max(nodes_by_id) + 1 if nodes_by_id else 1


def _position_um(node: dict) -> np.ndarray:
    return np.array(
        [float(node["z"]) * VOXEL_SCALE_UM[0], float(node["y"]) * VOXEL_SCALE_UM[1], float(node["x"]) * VOXEL_SCALE_UM[2]],
        dtype=np.float64,
    )


class _FrameCache:
    """``t -> full-res frame`` with memoisation; ``None`` source disables intensity access."""

    def __init__(self, source: FrameSource | None) -> None:
        self.source = source
        self.frames: dict[int, np.ndarray] = {}

    def get(self, t: int) -> np.ndarray | None:
        if self.source is None:
            return None
        if t not in self.frames:
            self.frames[t] = self.source(int(t))
        return self.frames[t]


# --------------------------------------------------------------------------
# synthetic node refinement + DeepCenter gate
# --------------------------------------------------------------------------

def refine_synthetic_midpoint(
    cfg: PostprocessConfig,
    frames: _FrameCache,
    t: int,
    midpoint: tuple[float, float, float],
    stats: dict[str, int],
) -> tuple[float, float, float]:
    if not cfg.gap_refine_synthetic or frames.source is None:
        return midpoint
    try:
        frame = frames.get(t)
        z, y, x = [int(round(v)) for v in midpoint]
        z0 = max(0, z - cfg.gap_refine_win_z)
        z1 = min(frame.shape[0], z + cfg.gap_refine_win_z + 1)
        y0 = max(0, y - cfg.gap_refine_win_yx)
        y1 = min(frame.shape[1], y + cfg.gap_refine_win_yx + 1)
        x0 = max(0, x - cfg.gap_refine_win_yx)
        x1 = min(frame.shape[2], x + cfg.gap_refine_win_yx + 1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        baseline = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - baseline, 0.0)
        total = float(weights.sum())
        if total <= 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        refined = (
            float((weights * zz).sum() / total),
            float((weights * yy).sum() / total),
            float((weights * xx).sum() / total),
        )
        if point_distance_um(refined, midpoint) > cfg.gap_refine_max_shift_um:
            stats["gap_refine_rejected_shift"] += 1
            return midpoint
        stats["gap_refined_synthetic"] += 1
        return refined
    except Exception:
        stats["gap_refine_failed"] += 1
        return midpoint


def deepcenter_accept_repair_point(
    cfg: PostprocessConfig,
    deepcenter: DeepCenter | None,
    frames: _FrameCache,
    heatmap_cache: dict,
    t: int,
    point: tuple[float, float, float],
    stats: dict[str, int],
    prefix: str,
    threshold: float,
) -> bool:
    if not cfg.use_deepcenter_veto:
        return True
    if deepcenter is None or frames.source is None:
        stats[f"deepcenter_{prefix}_missing"] += 1
        return True
    stats[f"deepcenter_{prefix}_checked"] += 1
    deepcenter.score_win_z = cfg.deepcenter_score_win_z
    deepcenter.score_win_yx = cfg.deepcenter_score_win_yx
    deepcenter.cache_max_frames = cfg.deepcenter_score_cache_max_frames
    deepcenter.tta = cfg.deepcenter_tta
    heatmap = deepcenter.heatmap(frames.get(t), heatmap_cache, int(t))
    score = deepcenter.score_point(heatmap, point)
    if score is None:
        stats[f"deepcenter_{prefix}_missing"] += 1
        return True
    if score < float(threshold):
        stats[f"deepcenter_{prefix}_rejected"] += 1
        return False
    stats[f"deepcenter_{prefix}_accepted"] += 1
    return True


# --------------------------------------------------------------------------
# passes
# --------------------------------------------------------------------------

def motion_relink_edges(
    cfg: PostprocessConfig,
    nodes_by_id: dict[int, dict],
    stats: dict[str, int],
    learned_edge_probs: dict[tuple[int, int], float] | None = None,
) -> list[dict]:
    if not cfg.output_motion_relink or not nodes_by_id:
        return []
    learned_edge_probs = learned_edge_probs or {}

    def learned_prob(source_id: int, target_id: int) -> float:
        value = learned_edge_probs.get((source_id, target_id), 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        return float(np.clip(value, 0.0, 1.0))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()

    frame_sizes = [len(ids) for ids in ids_by_t.values()]
    if frame_sizes and max(frame_sizes) > cfg.motion_relink_max_frame_nodes:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict] = []

    def assign_pass(source_ids: list[int], target_ids: list[int], gate_um: float):
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            if prev_pos is None:
                predicted = source_pos
            else:
                predicted = source_pos + cfg.motion_relink_velocity_weight * (source_pos - prev_pos)
            for j, target_id in enumerate(target_ids):
                target_pos = position_um[target_id]
                raw = float(np.linalg.norm(target_pos - source_pos))
                if raw > gate_um:
                    continue
                motion = float(np.linalg.norm(target_pos - predicted))
                prob = learned_prob(source_id, target_id)
                raw_dist[i, j] = raw
                motion_dist[i, j] = motion
                prob_matrix[i, j] = prob
                cost[i, j] = motion + 0.05 * raw - cfg.motion_relink_learned_bonus * prob
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= big:
                continue
            matches.append((
                source_ids[int(r)], target_ids[int(c)],
                float(raw_dist[r, c]), float(motion_dist[r, c]), float(prob_matrix[r, c]),
            ))
        return matches

    for t in sorted(ids_by_t):
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches = []
        for pass_name, gate_um in (("tight", cfg.motion_relink_tight_um), ("relaxed", cfg.motion_relink_relaxed_um)):
            pass_sources = [nid for nid in source_ids if nid in unmatched_sources]
            pass_targets = [nid for nid in target_ids if nid in unmatched_targets]
            for source_id, target_id, raw, motion, prob in assign_pass(pass_sources, pass_targets, gate_um):
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append((source_id, target_id, raw, motion, pass_name, prob))
                stats["motion_relink_tight_edges" if pass_name == "tight" else "motion_relink_relaxed_edges"] += 1
        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            selected_edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_prob": prob,
                "distance_um": raw,
                "motion_distance_um": motion,
                "motion_relinked": 1,
                "motion_pass": pass_name,
            })
            predecessor_position_um[target_id] = position_um[source_id]
        stats["motion_relink_frames"] += 1

    stats["motion_relink_edges"] = len(selected_edges)
    return selected_edges


def close_single_frame_gaps(
    cfg: PostprocessConfig,
    nodes_by_id: dict[int, dict],
    edges: list[dict],
    stats: dict[str, int],
    frames: _FrameCache,
    deepcenter: DeepCenter | None,
    heatmap_cache: dict,
) -> tuple[dict[int, dict], list[dict]]:
    if not cfg.output_gap_close or cfg.gap_close_max_gap < 1 or not edges:
        return nodes_by_id, edges

    outgoing = {int(e["source_id"]) for e in edges}
    incoming = {int(e["target_id"]) for e in edges}
    incident = outgoing | incoming

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    isolated_by_t: dict[int, list[int]] = {}
    all_ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        all_ids_by_t.setdefault(t, []).append(node_id)
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)
        if node_id not in incident:
            isolated_by_t.setdefault(t, []).append(node_id)

    max_synthetic = min(
        cfg.gap_close_max_added_abs,
        max(1, int(round(len(nodes_by_id) * cfg.gap_close_max_added_frac))) if cfg.gap_close_max_added_frac > 0 else 0,
    )
    next_id = _next_node_id(nodes_by_id)
    used_starts: set[int] = set()
    used_isolated: set[int] = set()
    synthetic_added = 0
    new_edges: list[dict] = []
    density_cache: dict[int, dict[int, float]] = {}

    def frame_local_spacing(t: int) -> dict[int, float]:
        cached = density_cache.get(t)
        if cached is not None:
            return cached
        frame_ids = all_ids_by_t.get(t, [])
        if len(frame_ids) <= 1:
            result = {nid: cfg.gap_density_reference_um for nid in frame_ids}
            density_cache[t] = result
            return result
        positions = np.stack([_position_um(nodes_by_id[nid]) for nid in frame_ids])
        tree = cKDTree(positions)
        query_k = min(len(frame_ids), max(2, cfg.gap_density_neighbors + 1))
        distances, _ = tree.query(positions, k=query_k)
        if distances.ndim == 1:
            distances = distances[:, None]
        result = {}
        for idx, nid in enumerate(frame_ids):
            nd = distances[idx, 1:]
            nd = nd[np.isfinite(nd)]
            result[nid] = float(np.median(nd)) if nd.size else cfg.gap_density_reference_um
        density_cache[t] = result
        stats["gap_density_nodes_scored"] += len(result)
        return result

    effective_gap_max = min(cfg.gap_close_max_gap, 1)
    stats["gap_close_effective_max_gap"] = effective_gap_max
    for gap in range(1, effective_gap_max + 1):
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = [sid for sid in starts_by_t.get(t + gap + 1, []) if sid not in used_starts]
            if not end_ids or not start_ids:
                continue
            end_points = [node_point(nodes_by_id[eid]) for eid in end_ids]
            start_points = [node_point(nodes_by_id[sid]) for sid in start_ids]
            threshold_um = cfg.gap_close_um * (gap + 1)
            d = np.zeros((len(end_ids), len(start_ids)), dtype=np.float64)
            adaptive_threshold = np.full_like(d, threshold_um)
            source_spacing = frame_local_spacing(t)
            target_spacing = frame_local_spacing(t + gap + 1)
            for i, ep in enumerate(end_points):
                for j, sp in enumerate(start_points):
                    d[i, j] = point_distance_um(ep, sp)
                    if cfg.gap_density_adaptive:
                        local_spacing = 0.5 * (
                            source_spacing.get(end_ids[i], cfg.gap_density_reference_um)
                            + target_spacing.get(start_ids[j], cfg.gap_density_reference_um)
                        )
                        step_delta = float(np.clip(
                            cfg.gap_density_gain * (local_spacing - cfg.gap_density_reference_um),
                            -cfg.gap_density_max_step_delta_um,
                            cfg.gap_density_max_step_delta_um,
                        ))
                        adaptive_threshold[i, j] = threshold_um + step_delta * (gap + 1)
                        stats["gap_density_step_delta_milli_sum"] += int(round(1000.0 * step_delta))

            base_allowed = d <= threshold_um
            adaptive_allowed = d <= adaptive_threshold
            stats["gap_density_candidates_expanded"] += int((adaptive_allowed & ~base_allowed).sum())
            stats["gap_density_candidates_restricted"] += int((base_allowed & ~adaptive_allowed).sum())
            stats["gap_candidates"] += int(adaptive_allowed.sum())
            if not np.isfinite(d).any():
                continue
            max_threshold = float(np.max(adaptive_threshold))
            big = max_threshold * 1000.0 + 1.0
            cost = np.where(adaptive_allowed, d, big)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if not adaptive_allowed[r, c]:
                    continue
                if not base_allowed[r, c]:
                    stats["gap_density_selected_outside_base"] += 1
                source_id = end_ids[int(r)]
                target_id = start_ids[int(c)]
                if source_id in outgoing or target_id in used_starts:
                    continue
                source = nodes_by_id[source_id]
                target = nodes_by_id[target_id]
                mid_t = int(source["t"]) + gap
                mid_point = (
                    (float(source["z"]) + float(target["z"])) / 2.0,
                    (float(source["y"]) + float(target["y"])) / 2.0,
                    (float(source["x"]) + float(target["x"])) / 2.0,
                )
                middle_id: int | None = None
                middle_reused = False
                if cfg.gap_close_reuse_existing:
                    candidates = [nid for nid in isolated_by_t.get(mid_t, []) if nid not in used_isolated]
                    if candidates:
                        distances = [point_distance_um(node_point(nodes_by_id[nid]), mid_point) for nid in candidates]
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= cfg.gap_close_reuse_um:
                            middle_id = candidates[best_idx]
                            middle_reused = True
                if middle_id is None:
                    if synthetic_added >= max_synthetic:
                        stats["gap_skipped_node_cap"] += 1
                        continue
                    middle_id = next_id
                    next_id += 1
                    refined_point = refine_synthetic_midpoint(cfg, frames, mid_t, mid_point, stats)
                    nodes_by_id[middle_id] = {
                        "node_id": middle_id, "t": mid_t,
                        "z": refined_point[0], "y": refined_point[1], "x": refined_point[2],
                        "gap_synthetic": 1,
                    }
                    synthetic_added += 1
                    stats["gap_inserted_synthetic"] += 1

                middle = nodes_by_id[middle_id]
                gap_span_um = float(d[r, c])
                marginal_gap = gap_span_um >= cfg.deepcenter_gap_confirm_min_span_um
                synthetic_middle = int(middle.get("gap_synthetic", 0)) == 1
                requires_center_confirmation = cfg.deepcenter_gap_veto and marginal_gap and synthetic_middle
                if cfg.deepcenter_gap_veto and not marginal_gap:
                    stats["deepcenter_gap_bypassed_strong_motion"] += 1
                elif cfg.deepcenter_gap_veto and not synthetic_middle:
                    stats["deepcenter_gap_bypassed_observed_node"] += 1
                if requires_center_confirmation and not deepcenter_accept_repair_point(
                    cfg, deepcenter, frames, heatmap_cache, mid_t, node_point(middle), stats, "gap", cfg.deepcenter_gap_threshold,
                ):
                    if int(middle.get("gap_synthetic", 0)) == 1:
                        nodes_by_id.pop(middle_id, None)
                        synthetic_added = max(0, synthetic_added - 1)
                        stats["gap_inserted_synthetic"] = max(0, stats["gap_inserted_synthetic"] - 1)
                    continue
                if middle_reused:
                    used_isolated.add(middle_id)
                    stats["gap_reused_existing"] += 1

                new_edges.append({"source_id": source_id, "target_id": middle_id, "edge_prob": None,
                                  "distance_um": edge_distance_um(source, middle), "gap_closed": 1})
                new_edges.append({"source_id": middle_id, "target_id": target_id, "edge_prob": None,
                                  "distance_um": edge_distance_um(middle, target), "gap_closed": 1})
                outgoing.add(source_id)
                incoming.add(middle_id)
                outgoing.add(middle_id)
                incoming.add(target_id)
                used_starts.add(target_id)
                stats["gap_pairs_selected"] += 1
                stats["gap_added_edges"] += 2

    if new_edges:
        edges = [*edges, *new_edges]
    stats["gap_added_nodes"] = stats["gap_inserted_synthetic"]
    return nodes_by_id, edges


def _single_successor_map(edges: list[dict]) -> dict[int, int]:
    by_source: dict[int, list[int]] = {}
    for e in edges:
        by_source.setdefault(int(e["source_id"]), []).append(int(e["target_id"]))
    return {s: t[0] for s, t in by_source.items() if len(t) == 1}


def _single_predecessor_map(edges: list[dict]) -> dict[int, int]:
    by_target: dict[int, list[int]] = {}
    for e in edges:
        by_target.setdefault(int(e["target_id"]), []).append(int(e["source_id"]))
    return {t: s[0] for t, s in by_target.items() if len(s) == 1}


def recover_strict_gap2(
    cfg: PostprocessConfig,
    nodes_by_id: dict[int, dict],
    edges: list[dict],
    stats: dict[str, int],
    frames: _FrameCache,
) -> tuple[dict[int, dict], list[dict]]:
    if not cfg.output_gap2_recovery or not edges or not nodes_by_id:
        return nodes_by_id, edges

    outgoing = {int(e["source_id"]) for e in edges}
    incoming = {int(e["target_id"]) for e in edges}
    predecessor = _single_predecessor_map(edges)
    successor = _single_successor_map(edges)

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)

    cap = min(cfg.gap2_max_links_abs, max(1, int(round(len(edges) * cfg.gap2_max_links_frac))))
    proposals: list[tuple[float, int, int, int, float]] = []
    scale = np.array(VOXEL_SCALE_UM)

    def pos_um(node_id: int) -> np.ndarray:
        n = nodes_by_id[node_id]
        return np.array([float(n["z"]), float(n["y"]), float(n["x"])], dtype=np.float64) * scale

    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = starts_by_t.get(t + 3, [])
        if not end_ids or not start_ids:
            continue
        for end_id in end_ids:
            end_pos = pos_um(end_id)
            for start_id in start_ids:
                start_pos = pos_um(start_id)
                dist = float(np.linalg.norm(start_pos - end_pos))
                if dist > cfg.gap2_max_total_um or dist / 3.0 > cfg.gap2_max_step_um:
                    continue
                step = (start_pos - end_pos) / 3.0
                context_penalty = 0.0
                if cfg.gap2_require_context:
                    ok_context = False
                    prev_id = predecessor.get(end_id)
                    if prev_id is not None:
                        prev_step = end_pos - pos_um(prev_id)
                        prev_norm = float(np.linalg.norm(prev_step))
                        step_norm = float(np.linalg.norm(step))
                        if prev_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(prev_step, step) / (prev_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(prev_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    nxt = successor.get(start_id)
                    if nxt is not None:
                        next_step = pos_um(nxt) - start_pos
                        next_norm = float(np.linalg.norm(next_step))
                        step_norm = float(np.linalg.norm(step))
                        if next_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(next_step, step) / (next_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(next_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    if not ok_context:
                        continue
                proposals.append((dist + 2.0 * context_penalty, end_id, start_id, t, dist))

    proposals.sort(key=lambda item: item[0])
    stats["gap2_candidates"] = len(proposals)
    if not proposals:
        return nodes_by_id, edges

    selected = []
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    per_frame_count: dict[int, int] = {}
    for proposal in proposals:
        if len(selected) >= cap:
            stats["gap2_skipped_cap"] += 1
            break
        _, end_id, start_id, t, _ = proposal
        if end_id in used_ends or start_id in used_starts:
            continue
        frame_cap = max(1, int(round(len(ends_by_t.get(t, [])) * cfg.gap2_frame_frac_cap)))
        if per_frame_count.get(t, 0) >= frame_cap:
            continue
        selected.append(proposal)
        used_ends.add(end_id)
        used_starts.add(start_id)
        per_frame_count[t] = per_frame_count.get(t, 0) + 1
    if not selected:
        return nodes_by_id, edges

    next_node_id = _next_node_id(nodes_by_id)
    local_frames = _FrameCache(frames.source)  # the notebook uses a fresh cache here
    new_edges: list[dict] = []
    for _, end_id, start_id, t, _ in selected:
        source = nodes_by_id[end_id]
        target = nodes_by_id[start_id]
        previous_id = end_id
        inserted = 0
        for k in (1, 2):
            frac = k / 3.0
            mid_t = int(source["t"]) + k
            midpoint = (
                float(source["z"]) + (float(target["z"]) - float(source["z"])) * frac,
                float(source["y"]) + (float(target["y"]) - float(source["y"])) * frac,
                float(source["x"]) + (float(target["x"]) - float(source["x"])) * frac,
            )
            refined_point = refine_synthetic_midpoint(cfg, local_frames, mid_t, midpoint, stats)
            node_id = next_node_id
            next_node_id += 1
            nodes_by_id[node_id] = {"node_id": node_id, "t": mid_t,
                                    "z": refined_point[0], "y": refined_point[1], "x": refined_point[2]}
            inserted += 1
            new_edges.append({"source_id": previous_id, "target_id": node_id, "edge_prob": None,
                              "distance_um": edge_distance_um(nodes_by_id[previous_id], nodes_by_id[node_id]),
                              "gap2_recovered": 1})
            previous_id = node_id
        new_edges.append({"source_id": previous_id, "target_id": start_id, "edge_prob": None,
                          "distance_um": edge_distance_um(nodes_by_id[previous_id], target), "gap2_recovered": 1})
        stats["gap2_pairs_selected"] += 1
        stats["gap2_added_nodes"] += inserted
        stats["gap2_added_edges"] += 3
    return nodes_by_id, [*edges, *new_edges]


def add_safe_divisions_postlink(
    cfg: PostprocessConfig,
    nodes_by_id: dict[int, dict],
    edges: list[dict],
    stats: dict[str, int],
    frames: _FrameCache,
    deepcenter: DeepCenter | None,
    heatmap_cache: dict,
) -> list[dict]:
    if not cfg.output_safe_divisions or not edges or not nodes_by_id:
        return edges

    out_by_source: dict[int, list[dict]] = {}
    incoming: set[int] = set()
    for e in edges:
        out_by_source.setdefault(int(e["source_id"]), []).append(e)
        incoming.add(int(e["target_id"]))
    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)

    existing_edges = {(int(e["source_id"]), int(e["target_id"])) for e in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * cfg.safe_div_global_frac_cap)))
    added: list[dict] = []
    used_targets: set[int] = set()

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [nid for nid in ids_by_t[t] if len(out_by_source.get(nid, [])) == 1]
        candidate_ids = [nid for nid in child_frame_ids if nid not in incoming and nid not in used_targets]
        if not source_ids or not candidate_ids:
            continue

        cpos = np.asarray([_position_um(nodes_by_id[c]) for c in candidate_ids], dtype=float)
        ctree = cKDTree(cpos) if len(cpos) else None

        def _succ1(nid):
            e = out_by_source.get(nid, [])
            return int(e[0]["target_id"]) if len(e) == 1 else None

        frame_cap = max(1, int(round(len(source_ids) * cfg.safe_div_frame_frac_cap)))
        proposals: list[tuple[float, int, int, float, float]] = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_id = int(out_by_source[source_id][0]["target_id"])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child["t"]) != t + 1:
                continue
            child_dist = edge_distance_um(source, existing_child)
            if child_dist > cfg.safe_div_existing_child_max_um:
                continue
            if source_id not in incoming:      # C1: parent must be mid-track
                continue
            if ctree is None:
                continue
            near = ctree.query_ball_point(_position_um(source), r=cfg.safe_div_max_um)
            mn_d, mn_i = ctree.query(_position_um(existing_child))
            mutual = candidate_ids[int(mn_i)] if mn_d <= cfg.safe_div_sister_max_um else None
            for ci in near:
                candidate_id = candidate_ids[int(ci)]
                if (source_id, candidate_id) in existing_edges:
                    continue
                if candidate_id != mutual:     # C2: mutual nearest orphans
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_distance_um(source, candidate)
                if parent_dist > cfg.safe_div_max_um:
                    continue
                sister_dist = edge_distance_um(existing_child, candidate)
                if sister_dist > cfg.safe_div_sister_max_um:
                    continue
                if cfg.deepcenter_safe_div_veto and not deepcenter_accept_repair_point(
                    cfg, deepcenter, frames, heatmap_cache, int(candidate["t"]), node_point(candidate),
                    stats, "safe_div", cfg.deepcenter_safe_div_threshold,
                ):
                    continue
                s1 = _succ1(existing_child_id)   # C3: divergence at t+2
                s2 = _succ1(candidate_id)
                if s1 is None or s2 is None:
                    continue
                n1 = nodes_by_id.get(s1)
                n2 = nodes_by_id.get(s2)
                if n1 is None or n2 is None:
                    continue
                if int(n1["t"]) != t + 2 or int(n2["t"]) != t + 2:
                    continue
                if edge_distance_um(n1, n2) - sister_dist < cfg.safe_div_diverge_um:
                    continue
                if cfg.safe_div_sister_symmetry_tau > 0.0:
                    sym_denom = max((child_dist + parent_dist) / 2.0, 1e-6)
                    if abs(child_dist - parent_dist) / sym_denom > cfg.safe_div_sister_symmetry_tau:
                        stats["safe_division_symmetry_rejected"] += 1
                        continue
                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))

        stats["safe_division_candidates"] += len(proposals)
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if len(added) >= global_cap:
                stats["safe_division_skipped_cap"] += 1
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            added.append({"source_id": source_id, "target_id": candidate_id, "edge_prob": None,
                          "distance_um": parent_dist, "safe_division": 1})
            used_targets.add(candidate_id)
            added_this_frame += 1

    if added:
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added]
    return edges


def filter_short_track_components(
    cfg: PostprocessConfig,
    nodes_by_id: dict[int, dict],
    edges: list[dict],
    stats: dict[str, int],
) -> tuple[dict[int, dict], list[dict]]:
    if not cfg.output_filter_short_tracks or cfg.output_min_track_len <= 1 or not edges:
        return nodes_by_id, edges

    parent = {nid: nid for nid in nodes_by_id}

    def find(nid: int) -> int:
        while parent[nid] != nid:
            parent[nid] = parent[parent[nid]]
            nid = parent[nid]
        return nid

    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    out_count: dict[int, int] = {}
    for e in edges:
        s, t = int(e["source_id"]), int(e["target_id"])
        union(s, t)
        out_count[s] = out_count.get(s, 0) + 1

    components: dict[int, list[int]] = {}
    for nid in nodes_by_id:
        components.setdefault(find(nid), []).append(nid)
    component_edges: dict[int, list[dict]] = {root: [] for root in components}
    for e in edges:
        s, t = int(e["source_id"]), int(e["target_id"])
        if s in parent and t in parent:
            component_edges.setdefault(find(s), []).append(e)

    keep: set[int] = set()
    for root, members in components.items():
        has_division = any(out_count.get(nid, 0) >= 2 for nid in members)
        if len(members) >= cfg.output_min_track_len or (cfg.output_keep_division_components and has_division):
            keep.update(members)
    if not keep:
        stats["short_track_filter_skipped_all"] += 1
        return nodes_by_id, edges

    removed_before_rescue = len(nodes_by_id) - len(keep)
    if removed_before_rescue <= 0:
        return nodes_by_id, edges

    if cfg.adaptive_short_track_rescue:
        removed_frac = removed_before_rescue / max(len(nodes_by_id), 1)
        if removed_frac >= cfg.short_track_rescue_trigger_removed_frac:
            budget = min(
                cfg.short_track_rescue_max_nodes_abs,
                max(0, int(round(len(nodes_by_id) * cfg.short_track_rescue_max_nodes_frac))),
            )
            stats["short_track_rescue_triggered"] = 1
            stats["short_track_rescue_budget"] = budget
            proposals = []
            for root, members in components.items():
                if set(members) & keep:
                    continue
                if len(members) < cfg.short_track_rescue_min_len or len(members) >= cfg.output_min_track_len:
                    continue
                c_edges = component_edges.get(root, [])
                if not c_edges:
                    continue
                probs, dists = [], []
                for e in c_edges:
                    try:
                        prob = float(e.get("edge_prob", 0.0))
                    except (TypeError, ValueError):
                        prob = 0.0
                    if np.isfinite(prob):
                        probs.append(prob)
                    try:
                        dist = float(e.get("distance_um", np.nan))
                    except (TypeError, ValueError):
                        dist = np.nan
                    if np.isfinite(dist):
                        dists.append(dist)
                mean_prob = float(np.mean(probs)) if probs else 0.0
                mean_dist = float(np.mean(dists)) if dists else float("inf")
                if mean_prob < cfg.short_track_rescue_min_mean_edge_prob:
                    continue
                if mean_dist > cfg.short_track_rescue_max_mean_edge_dist_um:
                    continue
                score = mean_prob - 0.02 * mean_dist + 0.004 * len(members)
                proposals.append((score, len(members), mean_prob, root, members))
            proposals.sort(reverse=True)
            rescued_nodes = rescued_components = 0
            for _, size, _, _, members in proposals:
                if budget <= 0 or rescued_nodes + size > budget:
                    continue
                keep.update(members)
                rescued_nodes += size
                rescued_components += 1
            stats["short_track_rescue_components"] = rescued_components
            stats["short_track_rescue_nodes"] = rescued_nodes

    removed_nodes = len(nodes_by_id) - len(keep)
    if removed_nodes <= 0:
        return nodes_by_id, edges
    kept_nodes = {nid: n for nid, n in nodes_by_id.items() if nid in keep}
    kept_edges = [e for e in edges if int(e["source_id"]) in kept_nodes and int(e["target_id"]) in kept_nodes]
    stats["short_track_components_removed"] = sum(1 for m in components.values() if not (set(m) & keep))
    stats["short_track_nodes_removed"] = removed_nodes
    stats["short_track_edges_removed"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


def linefit_smooth_output_graph(
    cfg: PostprocessConfig,
    nodes_by_id: dict[int, dict],
    edges: list[dict],
    stats: dict[str, int],
) -> dict[int, dict]:
    """Smooth linear track interiors without changing graph topology."""
    if not cfg.output_linefit_smooth or cfg.output_linefit_weight <= 0 or cfg.output_linefit_window <= 0 or not edges:
        return nodes_by_id

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for e in edges:
        s, t = int(e["source_id"]), int(e["target_id"])
        source, target = nodes_by_id.get(s), nodes_by_id.get(t)
        if source is None or target is None:
            continue
        if int(target["t"]) != int(source["t"]) + 1:
            continue
        successor.setdefault(s, []).append(t)
        predecessor.setdefault(t, []).append(s)

    original_pos = {nid: np.array([float(n["z"]), float(n["y"]), float(n["x"])], dtype=np.float64) for nid, n in nodes_by_id.items()}
    updated_pos: dict[int, np.ndarray] = {}
    weight = float(np.clip(cfg.output_linefit_weight, 0.0, 1.0))

    for node_id in sorted(nodes_by_id):
        neighbourhood = [(0, node_id)]
        current = node_id
        for step in range(1, cfg.output_linefit_window + 1):
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-step, current))
        current = node_id
        for step in range(1, cfg.output_linefit_window + 1):
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((step, current))
        if len(neighbourhood) < 3:
            stats["linefit_skipped_nodes"] += 1
            continue
        dts = np.array([delta for delta, _ in neighbourhood], dtype=np.float64)
        coords = np.stack([original_pos[nid] for _, nid in neighbourhood])
        fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)], dtype=np.float64)
        if not np.isfinite(fitted).all():
            stats["linefit_skipped_nodes"] += 1
            continue
        updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted

    for nid, pos in updated_pos.items():
        nodes_by_id[nid]["z"] = float(pos[0])
        nodes_by_id[nid]["y"] = float(pos[1])
        nodes_by_id[nid]["x"] = float(pos[2])
    stats["linefit_smoothed_nodes"] = len(updated_pos)
    return nodes_by_id


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

_STAT_KEYS = [
    "raw_edges", "dropped_nonconsecutive_edges", "dropped_long_edges", "dropped_multi_parent_edges",
    "dropped_multi_child_edges", "dropped_division_edges", "gap_candidates", "gap_pairs_selected",
    "gap_reused_existing", "gap_inserted_synthetic", "gap_added_nodes", "gap_added_edges",
    "gap_skipped_node_cap", "gap_density_nodes_scored", "gap_density_candidates_expanded",
    "gap_density_candidates_restricted", "gap_density_selected_outside_base",
    "gap_density_step_delta_milli_sum", "gap_refined_synthetic", "gap_refine_failed",
    "gap_refine_rejected_shift", "pruned_isolated_nodes", "motion_relink_edges",
    "motion_relink_tight_edges", "motion_relink_relaxed_edges", "motion_relink_frames",
    "motion_relink_replaced_raw_edges", "motion_relink_fallback_raw", "motion_relink_skipped_large_frame",
    "gap2_candidates", "gap2_pairs_selected", "gap2_added_nodes", "gap2_added_edges", "gap2_skipped_cap",
    "safe_division_candidates", "safe_divisions_added", "safe_division_skipped_cap",
    "safe_division_symmetry_rejected", "deepcenter_gap_checked", "deepcenter_gap_bypassed_strong_motion",
    "deepcenter_gap_bypassed_observed_node", "deepcenter_gap_accepted", "deepcenter_gap_rejected",
    "deepcenter_gap_missing", "deepcenter_safe_div_checked", "deepcenter_safe_div_accepted",
    "deepcenter_safe_div_rejected", "deepcenter_safe_div_missing", "short_track_components_removed",
    "short_track_nodes_removed", "short_track_edges_removed", "short_track_filter_skipped_all",
    "short_track_rescue_triggered", "short_track_rescue_components", "short_track_rescue_nodes",
    "short_track_rescue_budget", "linefit_smoothed_nodes", "linefit_skipped_nodes",
]


def filter_output_graph(
    nodes_by_id: dict[int, dict],
    raw_edges: list[dict],
    *,
    cfg: PostprocessConfig = PostprocessConfig(),
    frame_source: FrameSource | None = None,
    deepcenter: DeepCenter | None = None,
    name: str = "",
    verbose: bool = False,
) -> tuple[dict[int, dict], list[dict], dict[str, int]]:
    """Run every pass in the notebook's order. Mutates ``nodes_by_id`` and the edge dicts."""
    stats: dict[str, int] = {k: 0 for k in _STAT_KEYS}
    stats["raw_edges"] = len(raw_edges)
    log = print if verbose else (lambda *a, **k: None)

    edges: list[dict] = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge["source_id"]))
        target = nodes_by_id.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if cfg.output_enforce_next_frame and int(target["t"]) != int(source["t"]) + 1:
            stats["dropped_nonconsecutive_edges"] += 1
            continue
        distance_um = edge_distance_um(source, target)
        edge["distance_um"] = distance_um
        if cfg.output_edge_max_um > 0 and distance_um > cfg.output_edge_max_um:
            stats["dropped_long_edges"] += 1
            continue
        edges.append(edge)

    if cfg.output_motion_relink:
        learned: dict[tuple[int, int], float] = {}
        for edge in edges:
            prob = edge.get("edge_prob")
            if prob is None:
                continue
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            if np.isfinite(prob):
                key = (int(edge["source_id"]), int(edge["target_id"]))
                learned[key] = max(learned.get(key, float("-inf")), prob)
        motion_edges = motion_relink_edges(cfg, nodes_by_id, stats, learned)
        if motion_edges:
            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges
        else:
            stats["motion_relink_fallback_raw"] = 1

    if cfg.output_single_parent_repair and edges:
        best_by_target: dict[int, dict] = {}
        for edge in edges:
            tid = int(edge["target_id"])
            prev = best_by_target.get(tid)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_target[tid] = edge
        kept_ids = {id(e) for e in best_by_target.values()}
        stats["dropped_multi_parent_edges"] = sum(1 for e in edges if id(e) not in kept_ids)
        edges = [e for e in edges if id(e) in kept_ids]

    if cfg.output_single_child_repair and edges:
        best_by_source: dict[int, dict] = {}
        for edge in edges:
            sid = int(edge["source_id"])
            prev = best_by_source.get(sid)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_source[sid] = edge
        kept_ids = {id(e) for e in best_by_source.values()}
        stats["dropped_multi_child_edges"] = sum(1 for e in edges if id(e) not in kept_ids)
        edges = [e for e in edges if id(e) in kept_ids]

    log(f"  [{name}] after edge-filter+motion-relink: {len(nodes_by_id)} nodes, {len(edges)} edges")
    frames = _FrameCache(frame_source)
    heatmap_cache: dict = {}
    nodes_by_id, edges = close_single_frame_gaps(cfg, nodes_by_id, edges, stats, frames, deepcenter, heatmap_cache)
    nodes_by_id, edges = recover_strict_gap2(cfg, nodes_by_id, edges, stats, frames)
    log(f"  [{name}] after gap-closing (single-frame + gap2): {len(nodes_by_id)} nodes, {len(edges)} edges")
    edges = add_safe_divisions_postlink(cfg, nodes_by_id, edges, stats, frames, deepcenter, heatmap_cache)
    log(f"  [{name}] after safe-division repair: {len(nodes_by_id)} nodes, {len(edges)} edges "
        f"(candidates={stats['safe_division_candidates']}, added={stats['safe_divisions_added']})")

    if cfg.output_division_geometry_filter and edges:
        by_source: dict[int, list[dict]] = {}
        for edge in edges:
            by_source.setdefault(int(edge["source_id"]), []).append(edge)
        filtered: list[dict] = []
        for source_id, source_edges in by_source.items():
            if len(source_edges) <= 1:
                filtered.extend(source_edges)
                continue
            ranked = sorted(source_edges, key=edge_sort_key, reverse=True)
            source = nodes_by_id[source_id]
            top1, top2 = ranked[0], ranked[1]
            d1, d2 = float(top1["distance_um"]), float(top2["distance_um"])
            sister = edge_distance_um(nodes_by_id[int(top1["target_id"])], nodes_by_id[int(top2["target_id"])])
            valid_division = (
                max(d1, d2) <= cfg.div_parent_max_um
                and sister <= cfg.div_sister_max_um
                and int(nodes_by_id[int(top1["target_id"])]["t"]) == int(source["t"]) + 1
                and int(nodes_by_id[int(top2["target_id"])]["t"]) == int(source["t"]) + 1
            )
            if valid_division:
                filtered.extend([top1, top2])
                stats["dropped_division_edges"] += max(0, len(ranked) - 2)
            elif cfg.div_drop_to_single_if_bad:
                filtered.append(top1)
                stats["dropped_division_edges"] += len(ranked) - 1
            else:
                filtered.extend(ranked)
        edges = filtered

    if cfg.output_prune_isolated:
        incident = {int(e["source_id"]) for e in edges} | {int(e["target_id"]) for e in edges}
        if incident:
            kept_nodes = {nid: n for nid, n in nodes_by_id.items() if nid in incident}
            stats["pruned_isolated_nodes"] = len(nodes_by_id) - len(kept_nodes)
            nodes_by_id = kept_nodes
            edges = [e for e in edges if int(e["source_id"]) in nodes_by_id and int(e["target_id"]) in nodes_by_id]

    log(f"  [{name}] after prune-isolated: {len(nodes_by_id)} nodes, {len(edges)} edges")
    nodes_by_id, edges = filter_short_track_components(cfg, nodes_by_id, edges, stats)
    log(f"  [{name}] after short-track filtering: {len(nodes_by_id)} nodes, {len(edges)} edges "
        f"(components_removed={stats['short_track_components_removed']})")
    nodes_by_id = linefit_smooth_output_graph(cfg, nodes_by_id, edges, stats)
    log(f"  [{name}] FINAL: {len(nodes_by_id)} nodes, {len(edges)} edges")
    return nodes_by_id, edges, stats

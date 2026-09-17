"""Error decomposition for a predicted tracking graph against sparse ground truth.

`score_local.py` says *how much* is wrong; this says *why*. Every counted edge
error is assigned one cause, so the detector's share and the linker's share
can be read off separately, per embryo and per crowding level:

- FN (a GT edge we did not reproduce): an endpoint was never detected, or
  both were detected but linked elsewhere / left unlinked, or it is the
  second child of a division (structurally impossible without divisions).
- FP (a counted wrong predicted edge): the true partner WAS detected and the
  linker chose another detection (a pure association error), or it was not
  (nothing right to link to, so the root cause is the detector).

With a candidate dump (`--candidates`; NOTE: the ported pack pipeline has no
producer for these `.npz` files yet, so this path is currently unused) it also
reports the true partner's RANK among that source's distance-gated candidates
and the greedy-vs-exact-assignment REGRET on the same scores -- the two
numbers that decide whether the scorer or the selector is failing.

Reuses the metric's own node matching (`metric.match_nodes_per_frame`) so
every count reconciles exactly with `score_edges`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from cell_tracking.config import SCALE
from cell_tracking.io_geff import GeffGraph
from cell_tracking.io_geff import embryo_of, read_geff
from cell_tracking.metric import match_nodes_per_frame, score_edges

# Nearest-neighbour distance bins (um) for the crowding breakdown. From the
# measured distribution of detected nuclei on the held-out split: p5=6.0,
# median=8.9, p95=14.4 um.
CROWDING_BINS_UM: tuple[float, ...] = (7.0, 10.0, 14.0)


def crowding_bin(nn_um: float) -> str:
    edges = CROWDING_BINS_UM
    if nn_um < edges[0]:
        return f"<{edges[0]:g}"
    for lo, hi in zip(edges[:-1], edges[1:]):
        if nn_um < hi:
            return f"{lo:g}-{hi:g}"
    return f">={edges[-1]:g}"


@dataclass
class VolumeErrors:
    name: str
    embryo: str
    n_gt_nodes: int = 0
    n_gt_detected: int = 0
    n_gt_edges: int = 0
    tp: int = 0
    # --- FN causes (sum == n_gt_edges - tp) ---
    fn_undetected_both: int = 0
    fn_undetected_parent: int = 0
    fn_undetected_child: int = 0
    fn_wrong_link_both: int = 0
    fn_wrong_link_parent: int = 0
    fn_wrong_link_child: int = 0
    fn_unlinked: int = 0
    fn_division_second_child: int = 0
    # --- FP causes (sum == fp) ---
    fp: int = 0
    fp_partner_detected: int = 0
    fp_partner_undetected: int = 0
    fp_both_matched_wrong_pair: int = 0
    # --- association quality, conditioned on both endpoints detected ---
    assoc_edges: int = 0  # GT edges with both endpoints matched (excl. division 2nd child)
    assoc_tp: int = 0
    # rank of the true partner among the source's gated candidates (1 = top);
    # key "-1" = the true pair was not a candidate at all. Needs a dump.
    rank_hist: dict[str, int] = field(default_factory=dict)
    # TP(exact assignment on the same scores) - TP(greedy). Needs a dump.
    regret_tp: int | None = None
    regret_fp: int | None = None
    # per crowding bin of the source node's nearest neighbour: {bin: {gt_edges, tp, wrong_link}}
    by_crowding: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def fn(self) -> int:
        return self.n_gt_edges - self.tp

    @property
    def assoc_recall(self) -> float:
        return self.assoc_tp / self.assoc_edges if self.assoc_edges else float("nan")

    @property
    def det_recall(self) -> float:
        return self.n_gt_detected / self.n_gt_nodes if self.n_gt_nodes else float("nan")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(fn=self.fn, assoc_recall=self.assoc_recall, det_recall=self.det_recall)
        return d


def _positions_um(graph: GeffGraph) -> dict[int, np.ndarray]:
    return {
        int(i): np.array([z, y, x], dtype=np.float64) * SCALE
        for i, z, y, x in zip(graph.node_ids, graph.z, graph.y, graph.x)
    }


def load_candidates(npz_path: Path) -> dict | None:
    """The per-volume dump written by `predict_volume(dump_candidates=...)`, or None."""
    if not Path(npz_path).exists():
        return None
    with np.load(npz_path) as z:
        return {k: z[k] for k in z.files}


def decompose_volume(
    gt: GeffGraph, pred: GeffGraph, candidates: dict | None = None
) -> VolumeErrors:
    """Classify every counted edge error of one volume by cause."""
    name = getattr(pred, "name", None) or getattr(gt, "name", None) or ""
    out = VolumeErrors(name=name, embryo=embryo_of(name) if name else "")

    gt_by_t, pred_by_t = gt.nodes_by_t(), pred.nodes_by_t()
    pred_to_gt = match_nodes_per_frame(gt_by_t, pred_by_t)
    gt_to_pred = {g: p for p, g in pred_to_gt.items()}
    gt_edges = {(int(u), int(v)) for u, v in gt.edges}
    pred_edges = [(int(u), int(v)) for u, v in pred.edges]

    # Reconcile with the metric's own numbers.
    ref = score_edges(gt_edges, pred_edges, pred_to_gt)
    out.n_gt_nodes, out.n_gt_detected, out.n_gt_edges = len(gt.node_ids), len(pred_to_gt), len(gt_edges)
    out.tp, out.fp = ref.tp, ref.fp

    pred_out = {u: v for u, v in pred_edges}          # any one child (membership tests)
    pred_children: dict[int, set[int]] = {}           # all children: predicted graphs may fork
    for u, v in pred_edges:
        pred_children.setdefault(u, set()).add(v)
    pred_in = {v: u for u, v in pred_edges}
    gt_children: dict[int, list[int]] = {}
    for u, v in gt_edges:
        gt_children.setdefault(u, []).append(v)
    gt_parent_of = {v: u for u, v in gt_edges}
    pred_t = pred.t_of()
    pred_um = _positions_um(pred)

    # Nearest-neighbour distance of every predicted node within its frame (crowding).
    nn_um: dict[int, float] = {}
    for t, nodes in pred_by_t.items():
        if len(nodes) < 2:
            for nid, _ in nodes:
                nn_um[nid] = float("inf")
            continue
        ids = [nid for nid, _ in nodes]
        P = np.array([pred_um[i] for i in ids])
        d, _ = cKDTree(P).query(P, k=2)
        for nid, dist in zip(ids, d[:, 1]):
            nn_um[nid] = float(dist)

    # Candidate lookup: frame t -> (local index of pred node -> row ids), scores.
    cand_index: dict[int, dict] = {}
    if candidates is not None:
        n_t = int(candidates["n_t"])
        for t in range(n_t - 1):
            pairs = candidates.get(f"{t}:pairs")
            if pairs is None:
                continue
            ids_src = candidates[f"{t}:node_ids"]
            ids_dst = candidates[f"{t + 1}:node_ids"]
            cand_index[t] = {
                "pairs": pairs,
                "scores": candidates[f"{t}:scores"],
                "src_local": {int(n): i for i, n in enumerate(ids_src)},
                "dst_local": {int(n): i for i, n in enumerate(ids_dst)},
            }

    def bump(bin_key: str, field_: str) -> None:
        b = out.by_crowding.setdefault(bin_key, {"gt_edges": 0, "tp": 0, "wrong_link": 0, "undetected": 0})
        b[field_] += 1

    rank_hist: Counter = Counter()
    for u, v in gt_edges:
        pu, pv = gt_to_pred.get(u), gt_to_pred.get(v)
        is_div_child = len(gt_children[u]) >= 2
        bin_key = crowding_bin(nn_um.get(pu, float("inf"))) if pu is not None else "undetected-src"
        bump(bin_key, "gt_edges")
        if pu is not None and pv is not None and pv in pred_children.get(pu, ()):
            bump(bin_key, "tp")
            if not is_div_child:
                out.assoc_edges += 1
                out.assoc_tp += 1
                if pu in _t_of_cand(cand_index, pred_t, pu):
                    rank_hist[_rank(cand_index[pred_t[pu]], pu, pv)] += 1
            continue
        if pu is None and pv is None:
            out.fn_undetected_both += 1
            bump(bin_key, "undetected")
        elif pu is None:
            out.fn_undetected_parent += 1
            bump(bin_key, "undetected")
        elif pv is None:
            out.fn_undetected_child += 1
            bump(bin_key, "undetected")
        elif is_div_child:
            out.fn_division_second_child += 1
        else:
            out.assoc_edges += 1
            bump(bin_key, "wrong_link")
            if pu in pred_out and pv in pred_in:
                out.fn_wrong_link_both += 1
            elif pu in pred_out:
                out.fn_wrong_link_parent += 1
            elif pv in pred_in:
                out.fn_wrong_link_child += 1
            else:
                out.fn_unlinked += 1
            if pu in _t_of_cand(cand_index, pred_t, pu):
                rank_hist[_rank(cand_index[pred_t[pu]], pu, pv)] += 1
    out.rank_hist = {str(k): int(n) for k, n in sorted(rank_hist.items())}

    gt_has_child = set(gt_children)
    gt_has_parent = set(gt_parent_of)
    for s, d in pred_edges:
        gs, gd = pred_to_gt.get(s), pred_to_gt.get(d)
        if gs is not None and gd is not None and (gs, gd) in gt_edges:
            continue
        contradicted = (gd is not None and gd in gt_has_parent) or (gs is not None and gs in gt_has_child)
        if not contradicted:
            continue
        if gs is not None and gd is not None:
            out.fp_both_matched_wrong_pair += 1
        elif gd is not None:
            true_parent = gt_parent_of[gd]
            if gt_to_pred.get(true_parent) is not None:
                out.fp_partner_detected += 1
            else:
                out.fp_partner_undetected += 1
        else:
            kids = gt_children[gs]
            if any(gt_to_pred.get(k) is not None for k in kids):
                out.fp_partner_detected += 1
            else:
                out.fp_partner_undetected += 1

    if candidates is not None:
        out.regret_tp, out.regret_fp = _regret(cand_index, pred_to_gt, gt_edges, gt_has_child, gt_has_parent)

    # Internal consistency: every FN and FP has exactly one cause.
    fn_sum = (
        out.fn_undetected_both + out.fn_undetected_parent + out.fn_undetected_child
        + out.fn_wrong_link_both + out.fn_wrong_link_parent + out.fn_wrong_link_child
        + out.fn_unlinked + out.fn_division_second_child
    )
    assert fn_sum == out.fn, (name, fn_sum, out.fn)
    fp_sum = out.fp_partner_detected + out.fp_partner_undetected + out.fp_both_matched_wrong_pair
    assert fp_sum == out.fp, (name, fp_sum, out.fp)
    return out


def _t_of_cand(cand_index: dict, pred_t: dict, pu: int) -> set:
    t = pred_t.get(pu)
    return {pu} if t in cand_index and pu in cand_index[t]["src_local"] else set()


def _rank(ci: dict, pu: int, pv: int) -> int:
    """1-based rank of (pu -> pv) among pu's candidates by score; -1 if not a candidate."""
    ls, ld = ci["src_local"][pu], ci["dst_local"].get(pv)
    rows = np.nonzero(ci["pairs"][:, 0] == ls)[0]
    if ld is None or not len(rows):
        return -1
    scores = ci["scores"][rows]
    hit = rows[ci["pairs"][rows, 1] == ld]
    if not len(hit):
        return -1
    return int((scores > ci["scores"][hit[0]]).sum()) + 1


def _greedy_one_to_one(pairs: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Score-sorted greedy selection with in/out-degree <= 1 (the old repo's linker)."""
    if not len(pairs):
        return np.zeros((0, 2), dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    used_s: set[int] = set()
    used_d: set[int] = set()
    out = []
    for k in order:
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        if i in used_s or j in used_d:
            continue
        used_s.add(i)
        used_d.add(j)
        out.append((i, j))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def _regret(cand_index, pred_to_gt, gt_edges, gt_has_child, gt_has_parent) -> tuple[int, int]:
    """TP/FP of exact per-frame assignment minus greedy, on identical candidate scores."""
    d_tp = d_fp = 0
    for t, ci in cand_index.items():
        pairs, scores = ci["pairs"], ci["scores"]
        if not len(pairs):
            continue
        src_ids = {i: n for n, i in ci["src_local"].items()}
        dst_ids = {i: n for n, i in ci["dst_local"].items()}
        n_s, n_d = len(src_ids), len(dst_ids)
        # positions are irrelevant when pairs+scores are given, so pass dummies
        keep = scores >= 0.5 if scores.max() <= 1.0 and scores.min() >= 0.0 else np.ones(len(scores), bool)
        greedy = _greedy_one_to_one(pairs[keep], scores[keep])
        cost = np.full((n_s, n_d), 1e6)
        cost[pairs[keep, 0], pairs[keep, 1]] = -scores[keep]
        r, c = linear_sum_assignment(cost)
        exact = np.array([(i, j) for i, j in zip(r, c) if cost[i, j] < 1e5], dtype=np.int64).reshape(-1, 2)

        def count(sel):
            tp = fp = 0
            for i, j in sel:
                s, d = src_ids[int(i)], dst_ids[int(j)]
                gs, gd = pred_to_gt.get(s), pred_to_gt.get(d)
                if gs is not None and gd is not None and (gs, gd) in gt_edges:
                    tp += 1
                elif (gd is not None and gd in gt_has_parent) or (gs is not None and gs in gt_has_child):
                    fp += 1
            return tp, fp

        g_tp, g_fp = count(greedy)
        e_tp, e_fp = count(exact)
        d_tp += e_tp - g_tp
        d_fp += e_fp - g_fp
    return d_tp, d_fp


def analyze_volume(name: str, train_dir: Path, geff_dir: Path, candidates_dir: Path | None) -> VolumeErrors:
    gt = read_geff(Path(train_dir) / f"{name}.geff")
    pred = read_geff(Path(geff_dir) / f"{name}.geff")
    gt.name = pred.name = name  # type: ignore[attr-defined]
    cand = load_candidates(Path(candidates_dir) / f"{name}.npz") if candidates_dir else None
    return decompose_volume(gt, pred, cand)


def aggregate(results: list[VolumeErrors]) -> dict:
    """Sum the integer fields; recompute the derived rates."""
    agg: dict = {}
    for r in results:
        for k, v in asdict(r).items():
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, int):
                agg[k] = agg.get(k, 0) + v
            elif k == "rank_hist":
                h = agg.setdefault("rank_hist", {})
                for kk, n in v.items():
                    h[kk] = h.get(kk, 0) + n
            elif k == "by_crowding":
                bc = agg.setdefault("by_crowding", {})
                for b, d in v.items():
                    dd = bc.setdefault(b, {})
                    for kk, n in d.items():
                        dd[kk] = dd.get(kk, 0) + n
    agg["n_volumes"] = len(results)
    agg["fn"] = agg["n_gt_edges"] - agg["tp"]
    agg["det_recall"] = agg["n_gt_detected"] / agg["n_gt_nodes"] if agg.get("n_gt_nodes") else float("nan")
    agg["assoc_recall"] = agg["assoc_tp"] / agg["assoc_edges"] if agg.get("assoc_edges") else float("nan")
    if any(r.regret_tp is None for r in results):
        agg["regret_tp"] = agg["regret_fp"] = None
    return agg

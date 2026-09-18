#!/usr/bin/env python3
"""Build the data behind the interactive track-error viewer (docs/error_viewer/).

    python scripts/error_viewer_build.py --volume 6bba_3abfe10a --volume 44b6_e57ff5c6 \
        --pred-dir dist/preds_val_ea_off_motion_relink --ilp-dir dist/preds_val_ddp_best_blend_ilp \
        --det-dir /projects/.../detmaps_ddp_best_blend --out-dir dist/error_viewer

Writes, per volume, ``frames/<vol>/sNN.jpg`` (sheets of 10 xy max-projections of the raw frames, stacked
vertically, per-video quantile normalised) and ``data/<vol>.json`` with, per frame: every annotated cell with its outcome, every predicted
node, every predicted link classified by the metric (TP / FP / ignored) and the above-threshold detection
candidates the ILP dropped; plus annotated tracks with a per-frame status for the timeline strip.

Outcome of an annotated cell at frame t (what the strip shows):
  link_ok     detected, and its link to the next frame is a TP (or it has no annotated child)
  link_wrong  detected, has an annotated child, but the pipeline linked it elsewhere / nowhere
  dropped     not detected, but the detection map has a local maximum > 0.965 within 7 um (the ILP dropped it)
  weak        not detected, best local maximum within 7 um between 0.1 and 0.965
  none        not detected, no local maximum within 7 um
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cell_tracking.config import SCALE  # noqa: E402
from cell_tracking.io_geff import read_geff  # noqa: E402
from cell_tracking.io_zarr import read_array_meta, read_attrs, read_volume  # noqa: E402
from cell_tracking.metric import match_nodes_per_frame  # noqa: E402
from cell_tracking.preprocess import pack_normalize, video_quantiles  # noqa: E402

G = 1.625
GRID = np.array([1.0, 4.0, 4.0])


def ball_offsets(r_cells: float) -> np.ndarray:
    n = int(np.ceil(r_cells))
    zz, yy, xx = np.mgrid[-n:n + 1, -n:n + 1, -n:n + 1]
    m = (zz ** 2 + yy ** 2 + xx ** 2) <= r_cells ** 2
    return np.stack([zz[m], yy[m], xx[m]], axis=1)


OFFS = ball_offsets(7.0 / G)


def gt_tracks(gt) -> dict[int, int]:
    """Assign a track id to every GT node: follow single-child chains; children of a division start new tracks."""
    children: dict[int, list[int]] = defaultdict(list)
    has_parent: set[int] = set()
    for u, v in gt.edges:
        children[int(u)].append(int(v))
        has_parent.add(int(v))
    t_of = gt.t_of()
    roots = sorted((n for n in map(int, gt.node_ids) if n not in has_parent), key=lambda n: (t_of[n], n))
    tid_of: dict[int, int] = {}
    next_tid = 0
    stack = [(r, None) for r in roots]
    while stack:
        n, tid = stack.pop()
        if tid is None:
            tid = next_tid
            next_tid += 1
        tid_of[n] = tid
        kids = children.get(n, [])
        if len(kids) == 1:
            stack.append((kids[0], tid))
        else:
            for k in kids:
                stack.append((k, None))
    return tid_of


def build_volume(name: str, train_dir: Path, pred_dir: Path, ilp_dir: Path, det_dir: Path, out_dir: Path, jpeg_q: int) -> dict:
    zarr_path = train_dir / f"{name}.zarr"
    shape, dtype = read_array_meta(zarr_path)
    T, Z, Y, X = shape
    q_lo, q_hi = video_quantiles(read_attrs(zarr_path))
    gt = read_geff(train_dir / f"{name}.geff")
    pred = read_geff(pred_dir / f"{name}.geff")
    ilp = read_geff(ilp_dir / f"{name}.geff")

    # --- frames -------------------------------------------------------------------------------------
    # sheets of SHEET frames stacked vertically (keeps the published file count small)
    fdir = out_dir / "frames" / name
    fdir.mkdir(parents=True, exist_ok=True)
    SHEET = 10
    sheet = None
    for t in range(T):
        raw = read_volume(zarr_path, t, shape, dtype)
        proj = pack_normalize(raw.max(axis=0), q_lo, q_hi)
        hi = max(float(np.percentile(proj, 99.7)), 1e-3)
        img = np.clip(proj / hi, 0, 1)
        img8 = (np.sqrt(img) * 255).astype(np.uint8)  # gamma 0.5 lifts dim cells
        if sheet is None:
            sheet = np.zeros((SHEET * Y, X), dtype=np.uint8)
        sheet[(t % SHEET) * Y:(t % SHEET + 1) * Y] = img8
        if t % SHEET == SHEET - 1 or t == T - 1:
            Image.fromarray(sheet).save(fdir / f"s{t // SHEET:02d}.jpg", quality=jpeg_q)
            sheet = None

    # --- matching and classification --------------------------------------------------------------
    gt_by_t, pred_by_t = gt.nodes_by_t(), pred.nodes_by_t()
    p2g = match_nodes_per_frame(gt_by_t, pred_by_t)
    g2p = {g: p for p, g in p2g.items()}
    gt_edges = {(int(u), int(v)) for u, v in gt.edges}
    gt_children: dict[int, list[int]] = defaultdict(list)
    gt_parent: dict[int, int] = {}
    for u, v in gt_edges:
        gt_children[u].append(v)
        gt_parent[v] = u
    has_child = set(gt_children)
    has_parent = set(gt_parent)
    pred_out: dict[int, list[int]] = defaultdict(list)
    pedge_cls: dict[tuple[int, int], int] = {}
    for u, v in pred.edges:
        u, v = int(u), int(v)
        pred_out[u].append(v)
        gu, gv = p2g.get(u), p2g.get(v)
        if gu is not None and gv is not None and (gu, gv) in gt_edges:
            pedge_cls[(u, v)] = 1
        elif (gu is not None and gu in has_child) or (gv is not None and gv in has_parent):
            pedge_cls[(u, v)] = 2
        else:
            pedge_cls[(u, v)] = 0
    tp_gt_edges = {(p2g[u], p2g[v]) for (u, v), c in pedge_cls.items() if c == 1}
    tid_of = gt_tracks(gt)
    gt_pos = {int(n): (float(z), float(y), float(x)) for n, z, y, x in zip(gt.node_ids, gt.z, gt.y, gt.x)}
    pred_pos = {int(n): (float(z), float(y), float(x)) for n, z, y, x in zip(pred.node_ids, pred.z, pred.y, pred.x)}
    pred_t = pred.t_of()
    ilp_by_t = ilp.nodes_by_t()

    frames = []
    status_by_node: dict[int, str] = {}
    for t in range(T):
        # detection map: local maxima, candidates the ILP dropped
        det_path = det_dir / name / f"t{t:03d}.npy"
        cands_dropped: list[list[float]] = []
        p = is_max = None
        if det_path.exists():
            p = np.squeeze(np.load(det_path).astype(np.float32))
            is_max = p == maximum_filter(p, size=3, mode="nearest")
            cz, cy, cx = np.nonzero(is_max & (p > 0.965))
            cand_um = np.stack([cz * G, cy * G, cx * G], axis=1)
            ilp_nodes = ilp_by_t.get(t, [])
            if len(ilp_nodes) and len(cand_um):
                ilp_um = np.array([c for _, c in ilp_nodes], dtype=np.float64) * SCALE
                d, _ = cKDTree(ilp_um).query(cand_um, k=1)
                keep = d > 3.25
            else:
                keep = np.ones(len(cand_um), dtype=bool)
            for i in np.nonzero(keep)[0]:
                cands_dropped.append([int(cz[i]), int(cy[i] * 4), int(cx[i] * 4), round(float(p[cz[i], cy[i], cx[i]]), 3)])

        gt_rows = []
        for gid, (z, y, x) in gt_by_t.get(t, []):
            gid = int(gid)
            pid = g2p.get(gid)
            if pid is not None:
                if gid in has_child:
                    status = "link_ok" if any((gid, c) in tp_gt_edges for c in gt_children[gid]) else "link_wrong"
                else:
                    status = "link_ok"
                best = None
            else:
                best = None
                if p is not None:
                    c = np.array([z, y, x], dtype=np.float64) * SCALE / G
                    pts = np.round(c).astype(int) + OFFS
                    ok = (pts[:, 0] >= 0) & (pts[:, 0] < p.shape[0]) & (pts[:, 1] >= 0) & (pts[:, 1] < p.shape[1]) & (pts[:, 2] >= 0) & (pts[:, 2] < p.shape[2])
                    pts = pts[ok]
                    vals = p[pts[:, 0], pts[:, 1], pts[:, 2]]
                    lm = is_max[pts[:, 0], pts[:, 1], pts[:, 2]]
                    best = float(vals[lm].max()) if lm.any() else 0.0
                status = "dropped" if (best or 0) > 0.965 else "weak" if (best or 0) > 0.1 else "none"
            status_by_node[gid] = status
            gt_rows.append([gid, int(z), int(y), int(x), tid_of[gid], status, -1 if pid is None else int(pid),
                            None if best is None else round(best, 3)])
        pred_rows = [[int(n), int(round(z)), int(round(y)), int(round(x)), p2g.get(int(n), -1)] for n, (z, y, x) in pred_by_t.get(t, [])]
        pedges = [[u, v, c] for (u, v), c in pedge_cls.items() if pred_t[u] == t]
        frames.append({"gt": gt_rows, "pred": pred_rows, "pedges": pedges, "dropped": cands_dropped})

    # --- tracks ------------------------------------------------------------------------------------
    by_tid: dict[int, list[int]] = defaultdict(list)
    for n, tid in tid_of.items():
        by_tid[tid].append(n)
    gt_t = gt.t_of()
    tracks = []
    for tid, nodes in by_tid.items():
        nodes.sort(key=lambda n: gt_t[n])
        stat = {gt_t[n]: status_by_node[n] for n in nodes}
        errs = [tt for tt, s in stat.items() if s != "link_ok"]
        tracks.append({
            "tid": tid, "start": gt_t[nodes[0]], "end": gt_t[nodes[-1]], "n": len(nodes),
            "errors": len(errs), "first_error": min(errs) if errs else None,
            "status": [stat.get(tt) for tt in range(T)],
            "parent_tid": tid_of.get(gt_parent.get(nodes[0]), None) if nodes[0] in gt_parent else None,
        })
    tracks.sort(key=lambda tr: (-tr["errors"], tr["start"]))

    counts = defaultdict(int)
    for f in frames:
        for r in f["gt"]:
            counts[r[5]] += 1
    n_fp = sum(1 for c in pedge_cls.values() if c == 2)
    data = {
        "name": name, "T": T, "shape": [Z, Y, X], "scale": [1.625, 0.40625, 0.40625], "sheet": 10,
        "frames": frames, "tracks": tracks,
        "gt_edges": sorted([u, v] for u, v in gt_edges),
        "summary": {"gt_nodes": len(gt.node_ids), "gt_edges": len(gt_edges), "pred_nodes": len(pred.node_ids),
                    "tp": sum(1 for c in pedge_cls.values() if c == 1), "fp": n_fp, "fn": len(gt_edges) - len(tp_gt_edges),
                    "status_counts": dict(counts), "dropped_candidates": sum(len(f["dropped"]) for f in frames)},
    }
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")))
    return data["summary"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", action="append", dest="volumes", required=True)
    ap.add_argument("--train-dir", type=Path, default=Path("data/biohub-cell-tracking-during-development/train"))
    ap.add_argument("--pred-dir", type=Path, default=Path("dist/preds_val_ea_off_motion_relink"))
    ap.add_argument("--ilp-dir", type=Path, default=Path("dist/preds_val_ddp_best_blend_ilp"))
    ap.add_argument("--det-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("dist/error_viewer"))
    ap.add_argument("--jpeg-quality", type=int, default=82)
    ap.add_argument("--score-json", type=Path, default=Path("dist/score_ea_off_motion_relink.json"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    scores = {r["name"]: r for r in json.loads(a.score_json.read_text())} if a.score_json.exists() else {}
    index = []
    for v in a.volumes:
        s = build_volume(v, a.train_dir, a.pred_dir, a.ilp_dir, a.det_dir, a.out_dir, a.jpeg_quality)
        sc = scores.get(v, {})
        index.append({"name": v, "embryo": v.split("_")[0], "score": sc.get("adjusted_edge_jaccard"), "det_recall": sc.get("det_recall"), **s})
        print(v, s, flush=True)
    (a.out_dir / "data" / "index.json").write_text(json.dumps(index, indent=1))
    print("wrote", a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

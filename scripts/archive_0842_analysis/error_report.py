#!/usr/bin/env python3
"""Thorough error analysis of one held-out prediction: global statistics + figures + qualitative panels.

    PYTHONPATH=src python scripts/error_report.py --geff-dir dist/preds_val_0842_dump \
        --candidates dist/preds_val_0842_dump/candidates --score-json dist/score_0842_dump.json \
        --out-dir reports/figures/2026-09-14-error-analysis-0842

Builds on `cell_tracking.analysis.decompose_volume` (every counted edge error gets one cause) and adds
per-node / per-edge / per-track tables so that recall can be plotted against crowding, depth, time,
displacement, local intensity and score margin, plus image crops of representative errors.
Writes `<out-dir>/*.png`, `<out-dir>/stats.json` and prints the markdown tables for the report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from cell_tracking.analysis import aggregate, decompose_volume, load_candidates  # noqa: E402
from cell_tracking.config import SCALE  # noqa: E402
from cell_tracking.io_geff import embryo_of, read_geff  # noqa: E402
from cell_tracking.io_zarr import read_array_meta, read_volume  # noqa: E402
from cell_tracking.metric import match_nodes_per_frame  # noqa: E402

# ----------------------------------------------------------------------------- style
C = {"44b6": "#eb6834", "6bba": "#2a78d6", "all": "#52514e"}
C_GT, C_PRED, C_TRUE_EDGE, C_PRED_EDGE = "#7CFC00", "#ff4040", "#7CFC00", "#ff4040"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight",
})
EMBRYOS = ("44b6", "6bba")


def _pct(a, b):
    return 100.0 * a / b if b else float("nan")


def _bin_label(edges, i):
    if i == 0:
        return f"<{edges[0]:g}"
    if i == len(edges):
        return f">={edges[-1]:g}"
    return f"{edges[i-1]:g}-{edges[i]:g}"


def _binned_rate(x, ok, edges):
    """(labels, rate%, n) of `ok` per bin of x."""
    idx = np.digitize(x, edges)
    labels, rates, ns = [], [], []
    for i in range(len(edges) + 1):
        m = idx == i
        labels.append(_bin_label(edges, i))
        ns.append(int(m.sum()))
        rates.append(_pct(ok[m].sum(), m.sum()))
    return labels, np.array(rates), np.array(ns)


def _rate_plot(ax, x, ok, emb, edges, title, xlabel, min_n=10):
    labels = None
    for e in EMBRYOS + ("all",):
        m = np.ones(len(x), bool) if e == "all" else (emb == e)
        if not m.any():
            continue
        labels, r, n = _binned_rate(x[m], ok[m], edges)
        r = np.where(n >= min_n, r, np.nan)
        ax.plot(range(len(labels)), r, "-o", color=C[e], lw=2 if e == "all" else 1.5, ms=5,
                label=f"{e} (n={int(m.sum())})", alpha=0.95 if e == "all" else 0.85)
        if e == "all":
            for i, (rv, nv) in enumerate(zip(r, n)):
                if np.isfinite(rv):
                    ax.annotate(f"n={nv}", (i, rv), textcoords="offset points", xytext=(0, -13), ha="center",
                                fontsize=7, color="#52514e")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 102)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("%")
    ax.legend(fontsize=7, loc="lower left")


# ----------------------------------------------------------------------------- per-volume tables
def gt_tracks(gt_edges: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Chains of GT edges; a division ends the parent track and starts one per child."""
    children = defaultdict(list)
    has_parent = set()
    for u, v in gt_edges:
        children[u].append(v)
        has_parent.add(v)
    starts = [u for u in children if u not in has_parent]
    for u, kids in children.items():
        if len(kids) >= 2:
            starts.extend(kids)
    tracks = []
    for s in starts:
        tr, cur = [], s
        while len(children.get(cur, [])) == 1:
            nxt = children[cur][0]
            tr.append((cur, nxt))
            cur = nxt
        if tr:
            tracks.append(tr)
    return tracks


def analyze(name, train_dir, geff_dir, cand_dir):
    gt = read_geff(train_dir / f"{name}.geff")
    pred = read_geff(geff_dir / f"{name}.geff")
    gt.name = pred.name = name
    cand = load_candidates(cand_dir / f"{name}.npz") if cand_dir else None
    ve = decompose_volume(gt, pred, cand)
    emb = embryo_of(name)

    gt_by_t, pred_by_t = gt.nodes_by_t(), pred.nodes_by_t()
    pred_to_gt = match_nodes_per_frame(gt_by_t, pred_by_t)
    gt_to_pred = {g: p for p, g in pred_to_gt.items()}
    gt_um = {int(i): np.array([z, y, x], float) * SCALE for i, z, y, x in zip(gt.node_ids, gt.z, gt.y, gt.x)}
    pred_um = {int(i): np.array([z, y, x], float) * SCALE for i, z, y, x in zip(pred.node_ids, pred.z, pred.y, pred.x)}
    gt_t, pred_t = gt.t_of(), pred.t_of()
    n_t = int(max(gt.t.max(), pred.t.max())) + 1

    # KD-trees of predictions per frame (crowding as the detector/linker sees it)
    trees = {}
    for t, nodes in pred_by_t.items():
        ids = [nid for nid, _ in nodes]
        trees[t] = (ids, cKDTree(np.array([pred_um[i] for i in ids])))

    def nn_pred(t, p_um, exclude=None, k=2):
        if t not in trees:
            return float("inf")
        ids, tree = trees[t]
        d, j = tree.query(p_um, k=min(k, len(ids)))
        d, j = np.atleast_1d(d), np.atleast_1d(j)
        for dd, jj in zip(d, j):
            if exclude is None or ids[jj] != exclude:
                return float(dd)
        return float("inf")

    # --- nodes
    nodes = []
    for i in range(len(gt.node_ids)):
        g = int(gt.node_ids[i]); t = int(gt.t[i])
        p = gt_to_pred.get(g)
        um = gt_um[g]
        nodes.append(dict(
            name=name, emb=emb, gid=g, t=t, z=int(gt.z[i]), y=int(gt.y[i]), x=int(gt.x[i]),
            z_um=float(um[0]), detected=p is not None,
            match_um=float(np.linalg.norm(pred_um[p] - um)) if p is not None else float("nan"),
            nearest_pred_um=nn_pred(t, um, k=1),  # unconstrained
            crowd_um=nn_pred(t, um, exclude=p, k=2),  # nearest *other* prediction
            n15=(len(trees[t][1].query_ball_point(um, 15.0)) if t in trees else 0) - (1 if p is not None else 0),
            t_frac=t / max(n_t - 1, 1),
        ))

    # --- candidates index
    cand_index = {}
    if cand is not None:
        for t in range(int(cand["n_t"]) - 1):
            pairs = cand.get(f"{t}:pairs")
            if pairs is None:
                continue
            cand_index[t] = dict(pairs=pairs, scores=cand[f"{t}:scores"],
                                 src_local={int(n): i for i, n in enumerate(cand[f"{t}:node_ids"])},
                                 dst_local={int(n): i for i, n in enumerate(cand[f"{t+1}:node_ids"])})

    def cand_info(pu, pv):
        """rank of true pair, its score, the best score, the chosen target's score, n candidates."""
        t = pred_t.get(pu)
        ci = cand_index.get(t)
        if ci is None or pu not in ci["src_local"]:
            return None
        ls = ci["src_local"][pu]
        rows = np.nonzero(ci["pairs"][:, 0] == ls)[0]
        if not len(rows):
            return dict(rank=-1, s_true=np.nan, s_best=np.nan, n_cand=0)
        sc = ci["scores"][rows]
        ld = ci["dst_local"].get(pv)
        hit = rows[ci["pairs"][rows, 1] == ld] if ld is not None else []
        if not len(hit):
            return dict(rank=-1, s_true=np.nan, s_best=float(sc.max()), n_cand=len(rows))
        s_true = float(ci["scores"][hit[0]])
        return dict(rank=int((sc > s_true).sum()) + 1, s_true=s_true, s_best=float(sc.max()), n_cand=len(rows))

    # --- edges
    gt_edges = {(int(u), int(v)) for u, v in gt.edges}
    pred_edges = [(int(u), int(v)) for u, v in pred.edges]
    pred_out = {u: v for u, v in pred_edges}
    pred_in = {v: u for u, v in pred_edges}
    gt_children = defaultdict(list)
    for u, v in gt_edges:
        gt_children[u].append(v)
    edges = []
    for u, v in gt_edges:
        pu, pv = gt_to_pred.get(u), gt_to_pred.get(v)
        is_div = len(gt_children[u]) >= 2
        disp = float(np.linalg.norm(gt_um[v] - gt_um[u]))
        if pu is not None and pv is not None and pred_out.get(pu) == pv:
            outcome = "tp"
        elif pu is None and pv is None:
            outcome = "undetected_both"
        elif pu is None:
            outcome = "undetected_parent"
        elif pv is None:
            outcome = "undetected_child"
        elif is_div:
            outcome = "division_second_child"
        elif pu in pred_out and pv in pred_in:
            outcome = "wrong_link_both"
        elif pu in pred_out:
            outcome = "wrong_link_parent"
        elif pv in pred_in:
            outcome = "wrong_link_child"
        else:
            outcome = "unlinked"
        rec = dict(name=name, emb=emb, u=u, v=v, t=gt_t[u], disp_um=disp, outcome=outcome, is_div=is_div,
                   both_detected=pu is not None and pv is not None,
                   crowd_um=nn_pred(gt_t[u], pred_um[pu], exclude=pu, k=2) if pu is not None else float("nan"),
                   z_um=float(gt_um[u][0]), pu=pu, pv=pv,
                   chosen=pred_out.get(pu) if pu is not None else None,
                   chosen_disp_um=float(np.linalg.norm(pred_um[pred_out[pu]] - pred_um[pu])) if pu in pred_out else np.nan)
        info = cand_info(pu, pv) if (pu is not None and pv is not None) else None
        rec.update(info or dict(rank=None, s_true=np.nan, s_best=np.nan, n_cand=None))
        if pu in pred_out and info is not None:
            ch = cand_info(pu, pred_out[pu])
            rec["s_chosen"] = ch["s_true"] if ch else np.nan
        else:
            rec["s_chosen"] = np.nan
        edges.append(rec)

    # --- counted FPs
    gt_parent_of = {v: u for u, v in gt_edges}
    fps = []
    for s, d in pred_edges:
        gs, gd = pred_to_gt.get(s), pred_to_gt.get(d)
        if gs is not None and gd is not None and (gs, gd) in gt_edges:
            continue
        contradicted = (gd is not None and gd in gt_parent_of) or (gs is not None and gs in gt_children)
        if not contradicted:
            continue
        if gs is not None and gd is not None:
            cat = "both_matched_wrong_pair"
        elif gd is not None:
            cat = "partner_detected" if gt_to_pred.get(gt_parent_of[gd]) is not None else "partner_undetected"
        else:
            cat = "partner_detected" if any(gt_to_pred.get(k) is not None for k in gt_children[gs]) else "partner_undetected"
        ch = cand_info(s, d)
        fps.append(dict(name=name, emb=emb, s=s, d=d, t=pred_t[s], cat=cat,
                        disp_um=float(np.linalg.norm(pred_um[d] - pred_um[s])),
                        score=ch["s_true"] if ch else np.nan))

    # --- tracks
    tp_set = {(e["u"], e["v"]) for e in edges if e["outcome"] == "tp"}
    tracks = []
    for tr in gt_tracks(gt_edges):
        ok = [e in tp_set for e in tr]
        run = best = 0
        for o in ok:
            run = run + 1 if o else 0
            best = max(best, run)
        tracks.append(dict(name=name, emb=emb, n_edges=len(tr), n_tp=int(sum(ok)), longest_run=best,
                           n_breaks=int(len(ok) - sum(ok)), full=all(ok)))

    # --- all predicted edge displacements, detections per frame
    pred_disp = np.array([np.linalg.norm(pred_um[v] - pred_um[u]) for u, v in pred_edges]) if pred_edges else np.zeros(0)
    per_frame = np.array([len(pred_by_t.get(t, [])) for t in range(n_t)])
    return dict(ve=ve, nodes=nodes, edges=edges, fps=fps, tracks=tracks, pred_disp=pred_disp,
                per_frame=per_frame, n_pred_edges=len(pred_edges), n_pred_nodes=len(pred.node_ids),
                gt=gt, pred=pred, pred_to_gt=pred_to_gt, gt_um=gt_um, pred_um=pred_um, cand=cand, cand_index=cand_index)


# ----------------------------------------------------------------------------- images
def _quantile999(zarr_path):
    meta = json.load(open(Path(zarr_path) / "zarr.json"))
    q = meta.get("attributes", {}).get("image_statistics", {}).get("quantiles", {})
    return float(q.get("0.999", 0)) or None


def add_intensity(nodes, name, train_dir):
    """Local intensity at every GT node: mean of a 3x5x5 voxel box / video 0.999-quantile, and
    the same relative to the frame median (an SNR proxy)."""
    zp = train_dir / f"{name}.zarr"
    shape, dtype = read_array_meta(zp)
    q999 = _quantile999(zp) or 1.0
    by_t = defaultdict(list)
    for n in nodes:
        by_t[n["t"]].append(n)
    for t, ns in by_t.items():
        vol = read_volume(zp, t, shape, dtype).astype(np.float32)
        med = float(np.median(vol[::4, ::8, ::8]))
        for n in ns:
            z, y, x = n["z"], n["y"], n["x"]
            box = vol[max(z-1, 0):z+2, max(y-2, 0):y+3, max(x-2, 0):x+3]
            n["intensity_rel"] = float(box.mean() / q999)
            n["snr"] = float(box.mean() / max(med, 1.0))


def crop_xy(vol, z, y, x, hz, hyx):
    z0, z1 = max(z - hz, 0), min(z + hz + 1, vol.shape[0])
    y0, y1 = max(y - hyx, 0), min(y + hyx + 1, vol.shape[1])
    x0, x1 = max(x - hyx, 0), min(x + hyx + 1, vol.shape[2])
    return vol[z0:z1, y0:y1, x0:x1].max(axis=0), (y0, x0), (z0, z1)


def draw_example(axes, res, train_dir, name, src_gid, title, hz=4, hyx=20, shape=None):
    """Two panels: frame t and t+1 max-projected over a z slab around the GT source node, with every GT
    node (green o), predicted node (red +) inside the slab, the GT edge (green) and predicted edge(s) (red)."""
    gt, pred, gt_um, pred_um = res["gt"], res["pred"], res["gt_um"], res["pred_um"]
    zp = train_dir / f"{name}.zarr"
    shape, dtype = read_array_meta(zp)
    gz, gy, gx = gt.coords_of()[src_gid]
    t = gt.t_of()[src_gid]
    z, y, x = int(round(gz)), int(round(gy)), int(round(gx))
    gt_c, pred_c = gt.coords_of(), pred.coords_of()
    gt_by_t, pred_by_t = gt.nodes_by_t(), pred.nodes_by_t()
    gt_out = defaultdict(list)
    for u, v in gt.edges:
        gt_out[int(u)].append(int(v))
    pred_out = defaultdict(list)
    for u, v in pred.edges:
        pred_out[int(u)].append(int(v))
    gt_to_pred = {g: p for p, g in res["pred_to_gt"].items()}

    vols = [read_volume(zp, tt, shape, dtype).astype(np.float32) for tt in (t, t + 1)]
    vmax = max(np.percentile(v, 99.8) for v in vols)
    for k, (ax, tt, vol) in enumerate(zip(axes, (t, t + 1), vols)):
        img, (y0, x0), (z0, z1) = crop_xy(vol, z, y, x, hz, hyx)
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, extent=(x0 - .5, x0 + img.shape[1] - .5, y0 + img.shape[0] - .5, y0 - .5))
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#c3c2b7")
        def inside(c):
            return z0 <= c[0] < z1 and y0 <= c[1] < y0 + img.shape[0] and x0 <= c[2] < x0 + img.shape[1]
        for nid, _ in pred_by_t.get(tt, []):
            c = pred_c[nid]
            if inside(c):
                ax.plot(c[2], c[1], "+", color=C_PRED, ms=11, mew=1.8)
        for nid, _ in gt_by_t.get(tt, []):
            c = gt_c[nid]
            if inside(c):
                ax.plot(c[2], c[1], "o", mfc="none", mec=C_GT, ms=12, mew=1.6)
        if k == 1:  # edges from frame t into t+1, drawn on the t+1 panel
            for nid, _ in gt_by_t.get(t, []):
                c = gt_c[nid]
                if not inside((c[0], c[1], c[2])):
                    continue
                def mark_outside(d, ref, color, what="dz"):
                    # arrow endpoint lies outside the projected z slab: hollow square + dz label
                    if not (z0 <= d[0] < z1):
                        dz = (d[0] - ref[0]) * SCALE[0]
                        ax.plot(d[2], d[1], "s", mfc="none", mec=color, ms=12, mew=1.8, clip_on=True)
                        ax.annotate(f"{what} {dz:+.0f} µm", (d[2], d[1]), textcoords="offset points", xytext=(6, 6),
                                    fontsize=7, color=color, fontweight="bold", annotation_clip=True)
                for kid in gt_out.get(nid, []):
                    d = gt_c[kid]
                    ax.annotate("", xy=(d[2], d[1]), xytext=(c[2], c[1]), annotation_clip=False,
                                arrowprops=dict(arrowstyle="-|>", color=C_TRUE_EDGE, lw=2.2, mutation_scale=9, shrinkA=0, shrinkB=1))
                    mark_outside(d, c, C_TRUE_EDGE)
                p = gt_to_pred.get(nid)
                if p is not None:
                    for kid in pred_out.get(p, []):
                        d = pred_c[kid]
                        ax.annotate("", xy=(d[2], d[1]), xytext=(pred_c[p][2], pred_c[p][1]), annotation_clip=False,
                                    arrowprops=dict(arrowstyle="-|>", color=C_PRED_EDGE, lw=2.2, ls="--", mutation_scale=9, shrinkA=0, shrinkB=1))
                        mark_outside(d, pred_c[p], C_PRED_EDGE)
                        mark_outside(pred_c[p], c, C_PRED_EDGE, what="source detection dz")
            ax.plot(gx, gy, "o", mfc="none", mec="white", ms=20, mew=1.0, alpha=0.7)
        else:
            ax.plot(gx, gy, "o", mfc="none", mec="white", ms=20, mew=1.0, alpha=0.7)
        ax.set_xlim(x0 - .5, x0 + img.shape[1] - .5); ax.set_ylim(y0 + img.shape[0] - .5, y0 - .5)
        ax.set_title(f"t={tt}   z∈[{z0},{z1})", fontsize=8, loc="left")
        if k == 0:
            ax.annotate(f"{name}: {title}", (0, 1.0), xycoords="axes fraction", xytext=(0, 22), textcoords="offset points",
                        fontsize=7.5, va="bottom", ha="left", color="#0b0b0b", annotation_clip=False)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geff-dir", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, default=None)
    ap.add_argument("--score-json", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=Path("dist/heldout_split.json"))
    ap.add_argument("--train-dir", type=Path, default=Path("data/biohub-cell-tracking-during-development/train"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-images", action="store_true", help="skip zarr reads (intensity + qualitative panels)")
    ap.add_argument("--tag", default="0.842 checkpoint (detector_tmax30_epoch23.pt)")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    names = json.load(open(args.split))["datasets"]
    scores = {r["name"]: r for r in json.load(open(args.score_json))}

    results = {}
    for n in names:
        results[n] = analyze(n, args.train_dir, args.geff_dir, args.candidates)
        if not args.no_images:
            add_intensity(results[n]["nodes"], n, args.train_dir)
        print(f"analysed {n}: det {100*results[n]['ve'].det_recall:.1f}%  assoc {100*results[n]['ve'].assoc_recall:.1f}%", flush=True)

    nodes = [x for r in results.values() for x in r["nodes"]]
    edges = [x for r in results.values() for x in r["edges"]]
    fps = [x for r in results.values() for x in r["fps"]]
    tracks = [x for r in results.values() for x in r["tracks"]]
    ves = {e: aggregate([r["ve"] for n, r in results.items() if embryo_of(n) == e]) for e in EMBRYOS}
    ves["all"] = aggregate([r["ve"] for r in results.values()])
    stats = {"tag": args.tag, "aggregate": ves}

    def arr(recs, k, dtype=float):
        return np.array([x[k] for x in recs], dtype=dtype)

    n_emb, e_emb = arr(nodes, "emb", object), arr(edges, "emb", object)

    # ---------------------------------------------------------------- fig01: per-volume score
    fig, ax = plt.subplots(figsize=(10, 3.6))
    order = sorted(names, key=lambda n: (embryo_of(n), -scores[n]["adjusted_edge_jaccard"]))
    vals = [scores[n]["adjusted_edge_jaccard"] for n in order]
    cols = [C[embryo_of(n)] for n in order]
    ax.bar(range(len(order)), vals, color=cols, width=0.7)
    for i, n in enumerate(order):
        s = scores[n]
        ax.text(i, vals[i] + 0.01, f"{vals[i]:.2f}", ha="center", fontsize=7, color="#0b0b0b")
        ax.text(i, 0.02, f"n={s['n_gt_edges']}", ha="center", fontsize=6.5, color="white", rotation=90, va="bottom")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([n.split("_")[1] for n in order], rotation=60, ha="right", fontsize=7)
    for e in EMBRYOS:
        m = [i for i, n in enumerate(order) if embryo_of(n) == e]
        w = sum(scores[order[i]]["weight"] for i in m)
        sub = sum(scores[order[i]]["adjusted_edge_jaccard"] * scores[order[i]]["weight"] for i in m) / w
        ax.hlines(sub, min(m) - 0.4, max(m) + 0.4, color=C[e], lw=1.2, ls="--")
        ax.text(max(m) + 0.45, sub, f"{e} weighted {sub:.3f}", color=C[e], fontsize=8, va="center")
    tot = sum(scores[n]["adjusted_edge_jaccard"] * scores[n]["weight"] for n in names) / sum(scores[n]["weight"] for n in names)
    ax.axhline(tot, color=C["all"], lw=1, ls=":")
    ax.text(-0.4, tot + 0.012, f"total (weighted) {tot:.4f}", color=C["all"], fontsize=8, ha="left")
    ax.set_ylim(0, 1.0); ax.set_ylabel("adjusted edge Jaccard")
    ax.set_title(f"Per-volume held-out score, {args.tag}. Bars labelled with GT edge count (= metric weight).")
    fig.savefig(out / "fig01_per_volume_score.png"); plt.close(fig)

    # ---------------------------------------------------------------- fig02: error decomposition
    fn_keys = [("fn_undetected_both", "both endpoints undetected"), ("fn_undetected_parent", "parent undetected"),
               ("fn_undetected_child", "child undetected"), ("fn_wrong_link_both", "both detected, both linked elsewhere"),
               ("fn_wrong_link_parent", "parent linked to wrong child"), ("fn_wrong_link_child", "child linked to wrong parent"),
               ("fn_unlinked", "both detected, neither linked"), ("fn_division_second_child", "division 2nd child (no divisions modelled)")]
    fp_keys = [("fp_partner_detected", "true partner detected, linker chose another"),
               ("fp_partner_undetected", "true partner undetected"), ("fp_both_matched_wrong_pair", "both endpoints matched, wrong pair")]
    fn_cols = ["#0d366b", "#256abf", "#86b6ef", "#8a2f0e", "#d95926", "#f4a582", "#eda100", "#9b9a95"]
    fp_cols = ["#d95926", "#256abf", "#eda100"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for ax, keys, cols, kind in ((axes[0], fn_keys, fn_cols, "fn"), (axes[1], fp_keys, fp_cols, "fp")):
        rows = ["all", "44b6", "6bba"]
        for r_i, e in enumerate(rows):
            a = ves[e]; tot_ = a[kind]; left = 0
            for (k, lab), col in zip(keys, cols):
                v = a[k] / tot_ * 100 if tot_ else 0
                ax.barh(r_i, v, left=left, color=col, edgecolor="#fcfcfb", lw=1, label=lab if r_i == 0 else None)
                if v > 6:
                    ax.text(left + v / 2, r_i, f"{a[k]}", ha="center", va="center", fontsize=7,
                            color="white" if col not in ("#86b6ef", "#f4a582", "#eda100", "#9b9a95") else "#0b0b0b")
                left += v
            ax.text(101, r_i, f"n={tot_}", va="center", fontsize=8, color="#52514e")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows); ax.invert_yaxis()
        ax.set_xlim(0, 112); ax.set_xlabel("% of counted " + kind.upper())
        ax.set_title(f"{kind.upper()} by cause" + (" (GT edges we missed)" if kind == "fn" else " (counted wrong predicted edges)"))
        ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2 if kind == "fn" else 1)
        ax.grid(False)
    fig.savefig(out / "fig02_error_decomposition.png"); plt.close(fig)

    # ---------------------------------------------------------------- fig03: recall vs crowding
    e_ok = arr(edges, "outcome", object) == "tp"
    both = arr(edges, "both_detected", bool) & ~arr(edges, "is_div", bool)
    crowd_e = arr(edges, "crowd_um")
    n_ok, crowd_n = arr(nodes, "detected", bool), arr(nodes, "crowd_um")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    _rate_plot(axes[0], crowd_e[both], e_ok[both], e_emb[both], (5, 6, 7, 8, 10, 14),
               "Association recall vs. crowding\n(GT edges with both endpoints detected)", "nearest other detection to the source, µm")
    _rate_plot(axes[1], arr(nodes, "n15"), n_ok, n_emb, (2, 4, 6, 8, 11),
               "Detection recall vs. local density\n(all GT nodes)", "other detections within 15 µm of the GT node")
    stats["det_recall_by_density"] = {e: dict(zip(*_binned_rate(arr(nodes, "n15")[(n_emb == e) if e != "all" else slice(None)],
                                                                   n_ok[(n_emb == e) if e != "all" else slice(None)], (2, 4, 6, 8, 11))[:2]))
                                       for e in EMBRYOS + ("all",)}
    # candidates per source
    nc = np.array([x["n_cand"] if x["n_cand"] is not None else -1 for x in edges])
    m = both & (nc >= 0)
    _rate_plot(axes[2], nc[m].astype(float), e_ok[m], e_emb[m], (2, 3, 5, 8, 12),
               "Association recall vs. number of gated candidates\n(15 µm gate)", "candidates for the source")
    fig.savefig(out / "fig03_recall_vs_crowding.png"); plt.close(fig)
    stats["assoc_recall_by_crowding"] = {}
    for e in EMBRYOS + ("all",):
        m = both & ((e_emb == e) if e != "all" else True)
        labels, r, n = _binned_rate(crowd_e[m], e_ok[m], (5, 6, 7, 8, 10, 14))
        stats["assoc_recall_by_crowding"][e] = {lab: dict(recall=float(rv), n=int(nv)) for lab, rv, nv in zip(labels, r, n)}

    # ---------------------------------------------------------------- fig04: detection vs depth / time / intensity, localisation
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    _rate_plot(axes[0], arr(nodes, "z_um"), n_ok, n_emb, (20, 40, 60, 80), "Detection recall vs. depth (z)", "z of GT node, µm")
    _rate_plot(axes[1], arr(nodes, "t_frac") * 100, n_ok, n_emb, (20, 40, 60, 80), "Detection recall vs. time in video", "% of video")
    if "snr" in nodes[0]:
        snr = arr(nodes, "snr")
        _rate_plot(axes[2], snr, n_ok, n_emb, (2, 3, 4, 6, 8), "Detection recall vs. local intensity\n(3x5x5 box mean / frame median)", "intensity / frame median")
        stats["snr_median_detected"] = float(np.median(snr[n_ok])); stats["snr_median_missed"] = float(np.median(snr[~n_ok])) if (~n_ok).any() else None
    def store_rate(key, x, edges):
        stats[key] = {}
        for e in EMBRYOS + ("all",):
            m = (n_emb == e) if e != "all" else np.ones(len(x), bool)
            labels, r, n = _binned_rate(x[m], n_ok[m], edges)
            stats[key][e] = {lab: dict(recall=float(rv), n=int(nv)) for lab, rv, nv in zip(labels, r, n)}
    store_rate("det_recall_by_depth", arr(nodes, "z_um"), (20, 40, 60, 80))
    store_rate("det_recall_by_time", arr(nodes, "t_frac") * 100, (20, 40, 60, 80))
    if "snr" in nodes[0]:
        store_rate("det_recall_by_intensity", arr(nodes, "snr"), (2, 3, 4, 6, 8))
    ax = axes[3]
    mu = arr(nodes, "match_um")
    for e in EMBRYOS:
        m = (n_emb == e) & n_ok
        ax.hist(mu[m], bins=np.arange(0, 7.25, 0.25), histtype="step", lw=1.8, color=C[e], density=True,
                label=f"{e} median {np.median(mu[m]):.2f} µm")
    ax.axvline(1.625, color="#9b9a95", ls=":", lw=1); ax.text(1.7, ax.get_ylim()[1] * 0.9, "1 grid cell\n(1.625 µm)", fontsize=7, color="#52514e")
    ax.set_title("Localisation error of matched detections"); ax.set_xlabel("|pred - GT|, µm"); ax.set_ylabel("density"); ax.legend(fontsize=7)
    fig.savefig(out / "fig04_detection_depth_time_intensity_localisation.png"); plt.close(fig)
    stats["localisation_um"] = {e: dict(median=float(np.median(mu[(n_emb == e) & n_ok])), p90=float(np.percentile(mu[(n_emb == e) & n_ok], 90))) for e in EMBRYOS}
    missed = ~n_ok
    near = arr(nodes, "nearest_pred_um")
    stats["missed_nodes"] = {e: dict(n=int((missed & (n_emb == e)).sum()),
                                     within_7um_but_taken=int((missed & (n_emb == e) & (near <= 7)).sum()),
                                     nearest_7_10um=int((missed & (n_emb == e) & (near > 7) & (near <= 10)).sum()),
                                     nearest_gt_10um=int((missed & (n_emb == e) & (near > 10)).sum())) for e in EMBRYOS + ("all",) if e != "all"}
    stats["missed_nodes"]["all"] = dict(n=int(missed.sum()), within_7um_but_taken=int((missed & (near <= 7)).sum()),
                                        nearest_7_10um=int((missed & (near > 7) & (near <= 10)).sum()), nearest_gt_10um=int((missed & (near > 10)).sum()))

    # ---------------------------------------------------------------- fig05: rank, margin, displacement
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    ax = axes[0]
    ranks = np.array([x["rank"] if x["rank"] is not None else -2 for x in edges])
    for i, e in enumerate(EMBRYOS):
        m = both & (e_emb == e) & (ranks != -2)
        r = ranks[m]
        cats = ["1", "2", "3", ">=4", "not cand."]
        counts = [(r == 1).sum(), (r == 2).sum(), (r == 3).sum(), (r >= 4).sum(), (r == -1).sum()]
        pct = np.array(counts) / max(m.sum(), 1) * 100
        ax.bar(np.arange(5) + (i - 0.5) * 0.38, pct, width=0.36, color=C[e], label=f"{e} (n={m.sum()})")
        for j, (p, c) in enumerate(zip(pct, counts)):
            ax.text(j + (i - 0.5) * 0.38, p + 1, f"{p:.1f}%", ha="center", fontsize=6.5, rotation=0)
        stats[f"rank_pct_{e}"] = dict(zip(cats, map(float, pct)))
    ax.set_xticks(range(5)); ax.set_xticklabels(cats); ax.set_yscale("log"); ax.set_ylim(0.05, 200)
    ax.set_title("Rank of the true partner among the source's\ngated candidates (both endpoints detected)"); ax.set_ylabel("% (log)"); ax.legend(fontsize=7)
    ax = axes[1]
    s_true, s_chosen = arr(edges, "s_true"), arr(edges, "s_chosen")
    wl = both & np.isin(arr(edges, "outcome", object), ["wrong_link_both", "wrong_link_parent"]) & np.isfinite(s_true) & np.isfinite(s_chosen)
    margin = s_chosen[wl] - s_true[wl]
    neg = margin < 0
    ax.hist(margin[~neg], bins=np.linspace(-1.0, 1.0, 41), color="#d95926", alpha=0.9, label=f"true partner outscored (n={int((~neg).sum())})")
    ax.hist(margin[neg], bins=np.linspace(-1.0, 1.0, 41), color="#256abf", alpha=0.9, label=f"true partner scored higher but its target\nwas taken by a competing source (n={int(neg.sum())})")
    ax.axvline(0, color="#0b0b0b", lw=0.8)
    ax.set_title(f"Wrong links: score(chosen) − score(true partner)\n(n={wl.sum()})"); ax.set_xlabel("score margin"); ax.set_ylabel("edges"); ax.legend(fontsize=6.5, loc="upper left")
    wl_rank = ranks[wl]
    stats["wrong_link_margin"] = dict(n=int(wl.sum()), median=float(np.median(margin)), frac_negative=float(neg.mean()),
                                      frac_0_to_0_1=float(((margin >= 0) & (margin < 0.1)).mean()), frac_above_0_5=float((margin > 0.5).mean()),
                                      rank1_pct=float(100 * (wl_rank == 1).mean()), rank2_pct=float(100 * (wl_rank == 2).mean()),
                                      rank_ge3_pct=float(100 * (wl_rank >= 3).mean()), not_cand_pct=float(100 * (wl_rank == -1).mean()))
    for e in EMBRYOS:
        m = wl & (e_emb == e)
        stats["wrong_link_margin"][f"{e}_rank1_pct"] = float(100 * (ranks[m] == 1).mean())
        stats["wrong_link_margin"][f"{e}_n"] = int(m.sum())
    ax = axes[2]
    tpm = both & e_ok & np.isfinite(s_true)
    ax.hist(s_true[tpm], bins=np.linspace(0, 1, 26), color="#2a78d6", alpha=0.85, density=True, label=f"true partner, TP edges (n={tpm.sum()})")
    ax.hist(s_true[wl], bins=np.linspace(0, 1, 26), color="#eb6834", alpha=0.7, density=True, label=f"true partner, wrong-link edges (n={wl.sum()})")
    fsc = np.array([x["score"] for x in fps]); fsc = fsc[np.isfinite(fsc)]
    ax.hist(fsc, bins=np.linspace(0, 1, 26), histtype="step", color="#0b0b0b", lw=1.5, density=True, label=f"counted FP edges (n={len(fsc)})")
    ax.set_title("Edge-scorer output distributions"); ax.set_xlabel("pair score"); ax.set_ylabel("density"); ax.legend(fontsize=6.5)
    _rate_plot(axes[3], arr(edges, "disp_um")[both], e_ok[both], e_emb[both], (2, 4, 6, 8, 10, 15),
               "Association recall vs. GT displacement\n(both endpoints detected)", "GT displacement t→t+1, µm")
    fig.savefig(out / "fig05_rank_margin_scores_displacement.png"); plt.close(fig)

    wlm = np.isin(arr(edges, "outcome", object), ["wrong_link_both", "wrong_link_parent"])
    gd = arr(edges, "disp_um")
    stats["wrong_link_shorter_than_true_pct"] = float(100 * np.nanmean(arr(edges, "chosen_disp_um")[wlm] < gd[wlm]))
    # ---------------------------------------------------------------- fig06: displacement distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    gd = arr(edges, "disp_um")
    bins = np.arange(0, 20.5, 0.5)
    for e in EMBRYOS:
        axes[0].hist(gd[e_emb == e], bins=bins, histtype="step", lw=1.8, color=C[e], density=True, label=f"GT {e} (median {np.median(gd[e_emb == e]):.1f} µm, n={int((e_emb == e).sum())})")
        pd_ = np.concatenate([r["pred_disp"] for n, r in results.items() if embryo_of(n) == e])
        axes[0].hist(pd_, bins=bins, histtype="step", lw=1.2, ls="--", color=C[e], density=True, label=f"predicted {e} (median {np.median(pd_):.1f} µm, n={len(pd_)})")
    axes[0].set_title("Frame-to-frame displacement, GT vs. all predicted edges"); axes[0].set_xlabel("µm"); axes[0].set_ylabel("density"); axes[0].legend(fontsize=6.5)
    ax = axes[1]
    for e in EMBRYOS:
        m = (e_emb == e) & np.isin(arr(edges, "outcome", object), ["wrong_link_both", "wrong_link_parent"])
        cd = arr(edges, "chosen_disp_um")[m]
        ax.scatter(gd[m], cd, s=8, alpha=0.5, color=C[e], label=f"{e} wrong links (n={m.sum()})", edgecolor="none")
    lim = 22; ax.plot([0, lim], [0, lim], color="#9b9a95", lw=0.8, ls=":")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel("true displacement, µm"); ax.set_ylabel("displacement of the chosen (wrong) link, µm")
    ax.set_title(f"Wrong links: true vs. chosen displacement\n({stats['wrong_link_shorter_than_true_pct']:.0f}% of chosen links are the shorter hop)"); ax.legend(fontsize=7)
    fig.savefig(out / "fig06_displacement.png"); plt.close(fig)
    stats["gt_disp_um"] = {e: dict(median=float(np.median(gd[e_emb == e])), p90=float(np.percentile(gd[e_emb == e], 90)), p99=float(np.percentile(gd[e_emb == e], 99))) for e in EMBRYOS}

    # ---------------------------------------------------------------- fig07: tracks
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    t_emb = arr(tracks, "emb", object); n_e = arr(tracks, "n_edges"); full = arr(tracks, "full", bool)
    frac = arr(tracks, "n_tp") / n_e; lrun = arr(tracks, "longest_run") / n_e
    for e in EMBRYOS:
        m = t_emb == e
        axes[0].hist(n_e[m], bins=np.arange(0, 100, 4), histtype="step", lw=1.8, color=C[e], label=f"{e}: {m.sum()} tracks, median {np.median(n_e[m]):.0f} edges")
        axes[1].hist(frac[m] * 100, bins=np.linspace(0, 100, 21), histtype="step", lw=1.8, color=C[e], label=f"{e}: {100*full[m].mean():.0f}% fully recovered")
        xs = np.sort(lrun[m]); axes[2].step(xs * 100, np.arange(1, len(xs) + 1) / len(xs) * 100, where="post", color=C[e], lw=1.8, label=f"{e} (median {100*np.median(lrun[m]):.0f}%)")
        stats[f"tracks_{e}"] = dict(n=int(m.sum()), median_len=float(np.median(n_e[m])), full_pct=float(100 * full[m].mean()),
                                    weighted_edge_recall=float(100 * arr(tracks, "n_tp")[m].sum() / n_e[m].sum()),
                                    mean_breaks_per_100_edges=float(100 * arr(tracks, "n_breaks")[m].sum() / n_e[m].sum()))
    axes[0].set_title("GT track length (edges; split at divisions)"); axes[0].set_xlabel("edges"); axes[0].legend(fontsize=7)
    axes[1].set_title("Fraction of each GT track's edges recovered"); axes[1].set_xlabel("% edges TP"); axes[1].legend(fontsize=7, loc="upper left")
    axes[2].set_title("Longest unbroken correct segment (CDF)"); axes[2].set_xlabel("% of track length"); axes[2].set_ylabel("% of tracks"); axes[2].legend(fontsize=7)
    fig.savefig(out / "fig07_tracks.png"); plt.close(fig)

    # ---------------------------------------------------------------- fig08: node ratio / detections per frame
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    for n in order:
        pf = results[n]["per_frame"]
        ax.plot(np.arange(len(pf)), pf, color=C[embryo_of(n)], lw=0.9, alpha=0.75)
    ax.set_title("Detections per frame, every held-out volume (colour = embryo)"); ax.set_xlabel("frame t"); ax.set_ylabel("predicted nodes")
    ax.plot([], [], color=C["44b6"], label="44b6"); ax.plot([], [], color=C["6bba"], label="6bba"); ax.legend(fontsize=7)
    ax = axes[1]
    nr = [scores[n]["node_ratio"] for n in order]
    ax.bar(range(len(order)), nr, color=[C[embryo_of(n)] for n in order], width=0.7)
    ax.axhline(1.0, color="#0b0b0b", lw=0.8); ax.set_ylim(0.8, 1.5)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([n.split("_")[1] for n in order], rotation=60, ha="right", fontsize=7)
    ax.set_title("Node ratio: predicted nodes / estimated true nodes\n(>1 costs the adjusted Jaccard)"); ax.set_ylabel("ratio")
    fig.savefig(out / "fig08_detections_per_frame_node_ratio.png"); plt.close(fig)

    # ---------------------------------------------------------------- divisions
    n_div = sum(scores[n]["n_gt_divisions"] for n in names)
    stats["divisions"] = dict(gt=n_div, pred_forks=sum(scores[n]["pred_forks"] for n in names),
                              max_score_gain_if_perfect=0.1 * 1.0)

    # ---------------------------------------------------------------- qualitative
    if not args.no_images:
        rng = np.random.default_rng(0)
        E = np.array(edges, dtype=object)
        def pick(mask, key=None, k=1, reverse=False):
            idx = np.nonzero(mask)[0]
            if key is not None:
                idx = idx[np.argsort([key(edges[i]) for i in idx])]
                if reverse:
                    idx = idx[::-1]
            else:
                rng.shuffle(idx)
            return [edges[i] for i in idx[:k]]
        outc = arr(edges, "outcome", object)
        ex = []
        # linker: crowded 44b6 wrong link with small margin (hard), and a confident wrong link (large margin)
        m44 = both & (e_emb == "44b6") & np.isin(outc, ["wrong_link_both"]) & np.isfinite(s_chosen) & np.isfinite(s_true)
        ex += [("wrong link, 44b6, smallest score margin (near-tie)", x) for x in pick(m44, key=lambda x: x["s_chosen"] - x["s_true"], k=1)]
        ex += [("wrong link, 44b6, largest score margin (confident mistake)", x) for x in pick(m44, key=lambda x: x["s_chosen"] - x["s_true"], k=1, reverse=True)]
        m6 = both & (e_emb == "6bba") & np.isin(outc, ["wrong_link_both"]) & (crowd_e < 7)
        ex += [("wrong link, 6bba, crowded (<7 µm)", x) for x in pick(m6, k=1)]
        mlong = both & (outc == "wrong_link_both") & (gd > 10)
        ex += [("wrong link, fast-moving cell (GT displacement >10 µm)", x) for x in pick(mlong, key=lambda x: -x["disp_um"], k=1)]
        mtp = both & e_ok & (crowd_e < 6)
        ex += [("correct link in a crowded region (<6 µm), for contrast", x) for x in pick(mtp, k=1)]
        fig, axes = plt.subplots(len(ex), 2, figsize=(7.6, 4.1 * len(ex)))
        for row, (title, x) in zip(axes, ex):
            draw_example(row, results[x["name"]], args.train_dir, x["name"], x["u"],
                         f"{title}\nGT disp {x['disp_um']:.1f} µm; score true {x['s_true']:.2f} vs chosen {x['s_chosen']:.2f}; " + ("true partner beyond the 15 µm gate" if x['rank'] == -1 else f"true partner rank {x['rank']}") + f"; nearest other detection {x['crowd_um']:.1f} µm")
        fig.suptitle("Linker errors. green o = GT node, red + = detection, green arrow = GT edge, red dashed = predicted edge; white ring = the GT source in question;\nhollow square + dz = arrow endpoint outside the projected z slab (dz relative to the GT node); xy max-projection over ±4 z-slices (±6.5 µm) around it; crops are 41 x 41 voxels = 17 x 17 µm", fontsize=8, y=0.995)
        fig.subplots_adjust(hspace=0.45, wspace=0.08, top=0.94, bottom=0.01, left=0.03, right=0.99); fig.savefig(out / "fig09_qualitative_linker.png"); plt.close(fig)

        # detector: missed nodes -- lowest SNR, one crowded (nearest pred within 7 µm but taken), one isolated miss
        N = nodes
        n_out = arr(N, "emb", object)
        snr_ = arr(N, "snr") if "snr" in N[0] else np.full(len(N), np.nan)
        exn = []
        miss = ~n_ok
        def pickn(mask, key=None, k=1):
            idx = np.nonzero(mask)[0]
            idx = idx[np.argsort([key(N[i]) for i in idx])] if key else rng.permutation(idx)
            return [N[i] for i in idx[:k]]
        exn += [("missed detection, 6bba, dimmest (lowest intensity/median)", x) for x in pickn(miss & (n_out == "6bba"), key=lambda x: x["snr"])]
        exn += [("missed detection, 44b6, dimmest", x) for x in pickn(miss & (n_out == "44b6"), key=lambda x: x["snr"])]
        exn += [("missed detection, a detection within 7 µm but already matched to another GT node (merged pair)", x) for x in pickn(miss & (near <= 7) & (crowd_n < 6))]
        exn += [("missed detection, nothing predicted within 10 µm, bright cell", x) for x in pickn(miss & (near > 10), key=lambda x: -x["snr"])]
        fig, axes = plt.subplots(len(exn), 2, figsize=(7.6, 4.1 * len(exn)))
        for row, (title, x) in zip(axes, exn):
            draw_example(row, results[x["name"]], args.train_dir, x["name"], x["gid"],
                         f"{title}\nintensity/median {x['snr']:.1f}, nearest detection {x['nearest_pred_um']:.1f} µm, z={x['z']}")
        fig.suptitle("Detector errors. green o = GT node, red + = detection, green arrow = GT edge, red dashed = predicted edge; white ring = the missed GT node;\nhollow square + dz = arrow endpoint outside the projected z slab (dz relative to the GT node); xy max-projection over ±4 z-slices (±6.5 µm); crops are 41 x 41 voxels = 17 x 17 µm", fontsize=8, y=0.995)
        fig.subplots_adjust(hspace=0.45, wspace=0.08, top=0.93, bottom=0.01, left=0.03, right=0.99); fig.savefig(out / "fig10_qualitative_detector.png"); plt.close(fig)

        # overview frames: one per embryo, full xy max projection with all detections and GT
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        for ax, name in zip(axes, ("44b6_d5e7d891", "6bba_3abfe10a")):
            r = results[name]; zp = args.train_dir / f"{name}.zarr"; shape, dtype = read_array_meta(zp)
            t = 50; vol = read_volume(zp, t, shape, dtype).astype(np.float32)
            img = vol.max(axis=0)
            ax.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.8)); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
            pc, gc = r["pred"].coords_of(), r["gt"].coords_of()
            P = np.array([pc[n] for n, _ in r["pred"].nodes_by_t().get(t, [])]); G = np.array([gc[n] for n, _ in r["gt"].nodes_by_t().get(t, [])])
            if len(P): ax.plot(P[:, 2], P[:, 1], "+", color=C_PRED, ms=5, mew=0.8, alpha=0.9)
            if len(G): ax.plot(G[:, 2], G[:, 1], "o", mfc="none", mec=C_GT, ms=8, mew=1.2)
            ax.set_title(f"{name}  t={t}  full xy max-projection: {len(P)} detections (red +), {len(G)} GT nodes (green o); score {scores[name]['adjusted_edge_jaccard']:.2f}", fontsize=8)
        fig.tight_layout(); fig.savefig(out / "fig11_overview_frames.png"); plt.close(fig)

    # ---------------------------------------------------------------- stats dump + markdown
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=1, default=float)

    def row(label, f):
        return f"| {label} | " + " | ".join(f(ves[e]) for e in ("all",) + EMBRYOS) + " |"
    print("\n| | all 20 | 44b6 (5) | 6bba (15) |\n|---|---|---|---|")
    print(row("GT nodes / GT edges", lambda a: f"{a['n_gt_nodes']} / {a['n_gt_edges']}"))
    print(row("detection recall (7 µm bipartite)", lambda a: f"{100*a['det_recall']:.1f}%"))
    print(row("TP / FP / FN edges", lambda a: f"{a['tp']} / {a['fp']} / {a['fn']}"))
    print(row("association recall (both endpoints detected)", lambda a: f"{100*a['assoc_recall']:.1f}%"))
    det = lambda a: a["fn_undetected_both"] + a["fn_undetected_parent"] + a["fn_undetected_child"]
    link = lambda a: a["fn_wrong_link_both"] + a["fn_wrong_link_parent"] + a["fn_wrong_link_child"] + a["fn_unlinked"]
    print(row("FN: detector-attributable", lambda a: f"{det(a)} ({_pct(det(a), a['fn']):.1f}%)"))
    print(row("FN: linker-attributable", lambda a: f"{link(a)} ({_pct(link(a), a['fn']):.1f}%)"))
    print(row("FN: division second child", lambda a: f"{a['fn_division_second_child']} ({_pct(a['fn_division_second_child'], a['fn']):.1f}%)"))
    print(row("FP: partner detected, linker chose another", lambda a: f"{a['fp_partner_detected']} ({_pct(a['fp_partner_detected'], a['fp']):.1f}%)"))
    print(row("FP: partner undetected", lambda a: f"{a['fp_partner_undetected']} ({_pct(a['fp_partner_undetected'], a['fp']):.1f}%)"))
    print(row("FP: both matched, wrong pair", lambda a: f"{a['fp_both_matched_wrong_pair']} ({_pct(a['fp_both_matched_wrong_pair'], a['fp']):.1f}%)"))
    if ves["all"].get("regret_tp") is not None:
        print(row("greedy→exact assignment ΔTP / ΔFP", lambda a: f"{a['regret_tp']:+d} / {a['regret_fp']:+d}"))
    for e in EMBRYOS:
        print(f"rank {e}: {stats[f'rank_pct_{e}']}")
    print("assoc recall by crowding:", json.dumps(stats["assoc_recall_by_crowding"]))
    print("missed nodes:", json.dumps(stats["missed_nodes"]))
    print("det recall by density:", json.dumps(stats["det_recall_by_density"]))
    print("localisation:", json.dumps(stats["localisation_um"]))
    print("gt displacement:", json.dumps(stats["gt_disp_um"]))
    print("wrong-link margin/rank:", json.dumps(stats["wrong_link_margin"]), "shorter-than-true %:", stats["wrong_link_shorter_than_true_pct"])
    for e in EMBRYOS:
        print(f"tracks {e}:", json.dumps(stats[f"tracks_{e}"]))
    print("divisions:", stats["divisions"])
    if "snr_median_detected" in stats:
        print("snr median detected/missed:", stats["snr_median_detected"], stats["snr_median_missed"])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

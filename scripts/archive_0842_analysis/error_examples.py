#!/usr/bin/env python3
"""Frequency-ranked error categories of the 0.842 pipeline with typical examples (xy + xz side views).

    PYTHONPATH=src python scripts/error_examples_build.py          # tables.pkl (per miss / per linker FN)
    PYTHONPATH=src python scripts/error_examples_probe.py          # probe.pkl  (detector field at every miss, CPU)
    PYTHONPATH=src python scripts/error_examples.py --out-dir reports/figures/2026-09-16-error-examples

Every row of the two figures is one example of one category; categories are ranked by how many
counted errors they hold, and the example is the case closest to the category's median statistic
(not the most extreme one). Each panel marks the node in question (white ring), the detection nearest
to it (red ring) and, when that detection was matched to another GT node, the GT node that claimed it
(white line). The xz side view exposes z offsets that the xy projection hides.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_report import analyze  # noqa: E402

from cell_tracking.config import SCALE  # noqa: E402
from cell_tracking.io_zarr import read_array_meta, read_volume  # noqa: E402

TAU = 0.985
C_GT, C_PRED, C_FOCUS, C_CLAIM, C_TAKER = "#7CFC00", "#ff4040", "white", "#ffd23f", "#ff5fd7"
plt.rcParams.update({"figure.facecolor": "#0e0e0e", "axes.facecolor": "#0e0e0e", "text.color": "#f0f0f0",
                     "axes.edgecolor": "#555", "savefig.dpi": 150, "font.size": 8})

TRAIN = Path("data/biohub-cell-tracking-during-development/train")
GEFF = Path("dist/preds_val_0842_dump")
CAND = GEFF / "candidates"


# ----------------------------------------------------------------------------- categories
def categorize(tables, probe):
    M = pd.DataFrame(tables["misses"]); W = pd.DataFrame(tables["wrongs"]); N = pd.DataFrame(tables["nodes"])
    if probe:
        P = pd.DataFrame.from_dict(probe, orient="index").sort_index()
        M = pd.concat([M, P.reindex(range(len(M)))], axis=1)
    # local-metric truncation artefact: a detection within 7 um in float coords, unclaimed by any other GT node
    M["artefact"] = (M.near_um <= 7.0) & M.near_claimed_by.isna()
    M["merged"] = (M.near_um <= 7.0) & M.near_claimed_by.notna()
    M["bright"] = M.snr > 8
    M["dim"] = M.snr < 2.0
    M["run_cls"] = np.where(M.run_len == 1, "1 frame", np.where(M.run_len == 2, "2 frames", ">=3 frames"))
    if "p_max2" in M:
        # detector mechanism: field below threshold near the cell vs a >tau region that is not a kept peak (ridge to a neighbour)
        M["mech"] = np.where(M.artefact, "D5 local-metric truncation artefact (detection 5.6-7 um away, mostly in z)",
                    np.where(M.merged, "D4 merged: nearest detection claimed by a neighbouring GT cell",
                    np.where(M.p_max2 < TAU, np.where(M.run_len >= 3, "D1 sustained miss (>=3 frames), field below threshold",
                                                       "D2 transient miss (1-2 frames), field below threshold"),
                             "D3 field above threshold but no peak kept (ridge into a neighbour)")))
    else:
        M["mech"] = np.where(M.artefact, "D5 truncation artefact", np.where(M.merged, "D4 merged",
                    np.where(M.run_len >= 3, "D1 sustained miss", "D2 transient miss")))
    W["margin"] = W.s_chosen - W.s_true
    W["mech"] = np.where(W.outcome == "wrong_link_child", "L2 true target stolen by an unannotated competitor; cell left unlinked",
                np.where(W.outcome == "unlinked", "L3 true pair scored below 0.5; cell left unlinked",
                np.where(W.outcome == "division_second_child", "L4 division (second child never linked)",
                np.where(W["rank"] == -1, "L1d wrong partner: true partner beyond the 15 um gate",
                np.where(W["rank"] == 1, "L1c wrong partner: true partner ranked first but taken (greedy)",
                np.where(W.margin < 0.1, "L1a wrong partner chosen, near-tie (margin < 0.1)",
                np.where(W.margin > 0.5, "L1b wrong partner chosen, confident (margin > 0.5)",
                         "L1b wrong partner chosen, intermediate margin")))))))
    W["zjit"] = W.true_dz_um.abs() > 3
    return M, W, N


def md(df):
    """DataFrame -> markdown table (no tabulate dependency)."""
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    rows = [[("%.2f" % v if isinstance(v, float) else str(v)) for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def freq_tables(M, W, N):
    lines = []
    lines.append("### Detector: %d missed annotated cells (of %d; 44b6 %d, 6bba %d)\n" % (len(M), len(N), (M.emb == "44b6").sum(), (M.emb == "6bba").sum()))
    g = M.groupby("mech").agg(n=("gid", "size"), pct=("gid", lambda s: 100 * len(s) / len(M)), n44=("emb", lambda s: (s == "44b6").sum()),
                              snr_med=("snr", "median"), dim_pct=("dim", lambda s: 100 * s.mean()), bright_pct=("bright", lambda s: 100 * s.mean()),
                              near_med=("near_um", "median"), run_med=("run_len", "median"))
    if "p_max2" in M:
        g["p_max2_med"] = M.groupby("mech").p_max2.median()
    lines.append(md(g.round(2).sort_index()))
    lines.append("\nrun length x brightness (all misses):\n" + md(pd.crosstab(M.run_cls, pd.cut(M.snr, [0, 2, 4, 8, 100], labels=["<2 dim", "2-4", "4-8", ">8 bright"]), margins=True)))
    lines.append("\n### Linker: %d counted FN edges with both endpoints detected\n" % len(W))
    g = W.groupby("mech").agg(n=("u", "size"), pct=("u", lambda s: 100 * len(s) / len(W)), n44=("emb", lambda s: (s == "44b6").sum()),
                              s_true_med=("s_true", "median"), s_chosen_med=("s_chosen", "median"), crowd_med=("crowd_um", "median"),
                              gt_disp_med=("disp_um", "median"), det_dz_gt3_pct=("zjit", lambda s: 100 * s.mean()))
    lines.append(md(g.round(2).sort_index()))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- example selection
def typical(df, stat, k=2, rng=None, avoid_names=()):
    """k rows closest to the median of `stat`, distinct volumes, not in avoid_names."""
    if not len(df):
        return []
    med = df[stat].median()
    order = (df[stat] - med).abs().sort_values().index
    out, used = [], set(avoid_names)
    for i in order:
        if df.loc[i, "name"] in used:
            continue
        out.append(df.loc[i]); used.add(df.loc[i, "name"])
        if len(out) == k:
            break
    return out


# ----------------------------------------------------------------------------- drawing
_VOL_CACHE = {}


def get_volume(name, t):
    key = (name, t)
    if key not in _VOL_CACHE:
        zp = TRAIN / f"{name}.zarr"
        shape, dtype = read_array_meta(zp)
        if t < 0 or t >= shape[0]:
            return None
        _VOL_CACHE[key] = read_volume(zp, t, shape, dtype).astype(np.float32)
        if len(_VOL_CACHE) > 12:
            _VOL_CACHE.pop(next(iter(_VOL_CACHE)))
    return _VOL_CACHE[key]


def um(c):
    return np.asarray(c, float) * SCALE


class Scene:
    """One volume's graphs, indexed for drawing."""

    def __init__(self, name):
        self.name = name
        self.res = analyze(name, TRAIN, GEFF, CAND)
        gt, pred = self.res["gt"], self.res["pred"]
        self.gt_c, self.pred_c = gt.coords_of(), pred.coords_of()
        self.gt_t, self.pred_t = gt.t_of(), pred.t_of()
        self.gt_by_t, self.pred_by_t = defaultdict(list), defaultdict(list)
        for n, t in self.gt_t.items():
            self.gt_by_t[t].append(n)
        for n, t in self.pred_t.items():
            self.pred_by_t[t].append(n)
        self.p2g = self.res["pred_to_gt"]; self.g2p = {g: p for p, g in self.p2g.items()}
        self.gt_out, self.pred_out = defaultdict(list), {}
        for u, v in gt.edges:
            self.gt_out[int(u)].append(int(v))
        for u, v in pred.edges:
            self.pred_out[int(u)] = int(v)
        self.pred_in = {v: u for u, v in self.pred_out.items()}

    def score(self, pu, pv):
        ci = self.res["cand_index"].get(self.pred_t.get(pu))
        if ci is None or pu not in ci["src_local"] or pv not in ci["dst_local"]:
            return np.nan
        rows = np.nonzero((ci["pairs"][:, 0] == ci["src_local"][pu]) & (ci["pairs"][:, 1] == ci["dst_local"][pv]))[0]
        return float(ci["scores"][rows[0]]) if len(rows) else np.nan


def panel(ax, scene, t, center, view, focal, hz=4, hyx=28, side_hy=8, side_hz=10, window=None, title=""):
    """Draw one projection. view='xy': max over z in [cz-hz, cz+hz]; view='xz': max over y in [cy-side_hy, cy+side_hy].
    `focal` = dict(node_id -> style) for nodes that must be drawn even when outside the slab (hollow + offset label).
    Nodes are GT ('g', id) or pred ('p', id)."""
    vol = get_volume(scene.name, t)
    cz, cy, cx = [int(round(v)) for v in center]
    if vol is None:
        ax.set_axis_off(); ax.set_title(title + " (no frame)", loc="left", fontsize=7); return None
    Z, Y, X = vol.shape
    x0, x1 = max(cx - hyx, 0), min(cx + hyx + 1, X)
    if view == "xy":
        y0, y1 = max(cy - hyx, 0), min(cy + hyx + 1, Y)
        z0, z1 = max(cz - hz, 0), min(cz + hz + 1, Z)
        img = vol[z0:z1, y0:y1, x0:x1].max(axis=0)
        extent = (x0 - .5, x1 - .5, y1 - .5, y0 - .5)
        ax.imshow(img, cmap="gray", vmin=window[0], vmax=window[1], extent=extent, interpolation="nearest")
        in_slab = lambda c: z0 <= c[0] < z1
        in_win = lambda c: y0 <= c[1] < y1 and x0 <= c[2] < x1
        pos = lambda c: (c[2], c[1])
        off_label = lambda c: f"dz {(c[0] - center[0]) * SCALE[0]:+.0f}"
        ax.set_title(f"{title}  t={t}  xy, z∈[{z0},{z1})", loc="left", fontsize=7)
    else:
        y0, y1 = max(cy - side_hy, 0), min(cy + side_hy + 1, Y)
        z0, z1 = max(cz - side_hz, 0), min(cz + side_hz + 1, Z)
        img = vol[z0:z1, y0:y1, x0:x1].max(axis=1)
        extent = (x0 - .5, x1 - .5, z1 - .5, z0 - .5)
        ax.imshow(img, cmap="gray", vmin=window[0], vmax=window[1], extent=extent, interpolation="nearest", aspect=SCALE[0] / SCALE[2])
        in_slab = lambda c: y0 <= c[1] < y1
        in_win = lambda c: z0 <= c[0] < z1 and x0 <= c[2] < x1
        pos = lambda c: (c[2], c[0])
        off_label = lambda c: f"dy {(c[1] - center[1]) * SCALE[1]:+.0f}"
        ax.set_title(f"{title}  t={t}  xz side view, y∈[{y0},{y1})", loc="left", fontsize=7)
        ax.set_ylabel("z (slices)", fontsize=6)
    ax.set_xticks([]); ax.set_yticks([])
    for nid in scene.pred_by_t.get(t, []):
        c = scene.pred_c[nid]
        if in_win(c) and in_slab(c):
            ax.plot(*pos(c), "+", color=C_PRED, ms=9, mew=1.5)
    for nid in scene.gt_by_t.get(t, []):
        c = scene.gt_c[nid]
        if in_win(c) and in_slab(c):
            ax.plot(*pos(c), "o", mfc="none", mec=C_GT, ms=10, mew=1.4)
    for (kind, nid), st in focal.items():
        c = scene.gt_c.get(nid) if kind == "g" else scene.pred_c.get(nid)
        if c is None or (kind == "g" and scene.gt_t[nid] != t) or (kind == "p" and scene.pred_t[nid] != t):
            continue
        if not in_win(c):
            continue
        marker = "o" if st.get("shape", "ring") == "ring" else "s"
        if in_slab(c):
            ax.plot(*pos(c), marker, mfc="none", mec=st["color"], ms=st.get("ms", 15), mew=1.4, alpha=0.95)
        else:
            ax.plot(*pos(c), "s", mfc="none", mec=st["color"], ms=11, mew=1.4, alpha=0.95)
            ax.annotate(off_label(c) + " µm", pos(c), textcoords="offset points", xytext=(6, 5), fontsize=6, color=st["color"], fontweight="bold")
        if st.get("label"):
            ax.annotate(st["label"], pos(c), textcoords="offset points", xytext=(6, -11), fontsize=6, color=st["color"])
    return dict(pos=pos, in_win=in_win, in_slab=in_slab, lims=(extent[0], extent[1], extent[2], extent[3]))


def arrow(ax, geo, a, b, color, ls="-", lw=1.8):
    if geo is None or a is None or b is None:
        return
    pa, pb = geo["pos"](a), geo["pos"](b)
    ax.annotate("", xy=pb, xytext=pa, annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=8, shrinkA=0, shrinkB=1))


def line(ax, geo, a, b, color, label=None):
    if geo is None:
        return
    pa, pb = geo["pos"](a), geo["pos"](b)
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, lw=1.0, alpha=0.9)
    if label:
        ax.annotate(label, ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2), textcoords="offset points", xytext=(4, 4), fontsize=6, color=color)


def crop_window(scene, ts, center, hz=4, hyx=28):
    vals = []
    cz, cy, cx = [int(round(v)) for v in center]
    for t in ts:
        vol = get_volume(scene.name, t)
        if vol is None:
            continue
        sub = vol[max(cz - hz, 0):cz + hz + 1, max(cy - hyx, 0):cy + hyx + 1, max(cx - hyx, 0):cx + hyx + 1]
        vals.append(sub.ravel())
    v = np.concatenate(vals)
    return float(np.percentile(v, 55)), float(np.percentile(v, 99.8))


def fix_lims(axes_geo):
    for ax, geo in axes_geo:
        if geo is not None:
            l = geo["lims"]; ax.set_xlim(l[0], l[1]); ax.set_ylim(l[2], l[3])


# ----------------------------------------------------------------------------- detector rows
def draw_detector_row(axes, scene, m):
    g = int(m.gid); t = int(m.t); c = scene.gt_c[g]
    near = int(m.near_pid); claimer = None if pd.isna(m.near_claimed_by) else int(m.near_claimed_by)
    focal = {("g", g): dict(color=C_FOCUS, ms=16), ("p", near): dict(color=C_PRED, ms=13, label=f"nearest det {m.near_um:.1f} µm")}
    if claimer is not None:
        focal[("g", claimer)] = dict(color=C_CLAIM, ms=12, label=f"GT that claimed it ({m.near_claimed_um:.1f} µm)")
    win = crop_window(scene, (t - 1, t, t + 1), c)
    geos = []
    for ax, tt in zip(axes[:3], (t - 1, t, t + 1)):
        geo = panel(ax, scene, tt, c, "xy", focal if tt == t else {("g", g): dict(color=C_FOCUS, ms=16)}, window=win)
        geos.append((ax, geo))
        if geo is not None and tt != t:
            # the GT track through this cell: its parent / child node in that frame (white small ring) so the flicker is visible
            for nid in scene.gt_by_t.get(tt, []):
                if (tt == t + 1 and nid in scene.gt_out.get(g, [])) or (tt == t - 1 and g in scene.gt_out.get(nid, [])):
                    ax.plot(*geo["pos"](scene.gt_c[nid]), "o", mfc="none", mec=C_FOCUS, ms=16, mew=1.2, alpha=0.9)
                    p = scene.g2p.get(nid)
                    if p is not None:
                        ax.plot(*geo["pos"](scene.pred_c[p]), "o", mfc="none", mec=C_PRED, ms=12, mew=1.2, alpha=0.9)
                        ax.annotate("detected", geo["pos"](scene.gt_c[nid]), textcoords="offset points", xytext=(8, 8), fontsize=6, color=C_FOCUS)
                    else:
                        ax.annotate("also missed", geo["pos"](scene.gt_c[nid]), textcoords="offset points", xytext=(8, 8), fontsize=6, color=C_FOCUS)
    geo = panel(axes[3], scene, t, c, "xz", focal, window=win)
    geos.append((axes[3], geo))
    for ax, gg in ((axes[1], geos[1][1]), (axes[3], geo)):
        if gg is not None and claimer is not None:
            line(ax, gg, scene.pred_c[near], scene.gt_c[claimer], C_CLAIM)
    fix_lims(geos)


def detector_title(m):
    bits = [f"{m['name']}  t={int(m.t)}  z={int(m.z)}", f"brightness {m.snr:.1f}x frame median", f"missed for {int(m.run_len)} consecutive frame(s)",
            f"nearest detection {m.near_um:.1f} µm (xy {m.near_dxy_um:.1f}, z {m.near_dz_um:+.1f})"]
    if "p_max2" in m and not pd.isna(m.get("p_max2", np.nan)):
        bits.append(f"detector p at cell {m.p_at:.3f}, max within ±2 cells {m.p_max2:.3f} (τ = {TAU})")
    return "; ".join(bits)


# ----------------------------------------------------------------------------- linker rows
def draw_linker_row(axes, scene, w):
    u, v, t = int(w.u), int(w.v), int(w.t); pu, pv = int(w.pu), int(w.pv)
    ch = None if w.chosen is None or (isinstance(w.chosen, float) and np.isnan(w.chosen)) else int(w.chosen)
    taker = None if pd.isna(w.true_target_taken_by) else int(w.true_target_taken_by)
    c = scene.gt_c[u]
    focal_t = {("g", u): dict(color=C_FOCUS, ms=16), ("p", pu): dict(color=C_PRED, ms=12, label="source det")}
    focal_t1 = {("g", v): dict(color=C_FOCUS, ms=16, label="true target (GT)"), ("p", pv): dict(color=C_PRED, ms=12, label=f"true target det (score {w.s_true:.2f})")}
    if ch is not None and ch != pv:
        focal_t1[("p", ch)] = dict(color=C_CLAIM, ms=14, label=f"chosen (score {w.s_chosen:.2f})")
    if taker is not None and taker != pu:
        focal_t[("p", taker)] = dict(color=C_TAKER, ms=14, label=f"competitor (score {w.taker_score:.2f})")
    win = crop_window(scene, (t, t + 1), c)
    g0 = panel(axes[0], scene, t, c, "xy", focal_t, window=win)
    g1 = panel(axes[1], scene, t + 1, c, "xy", focal_t1, window=win)
    g2 = panel(axes[2], scene, t, c, "xz", focal_t, window=win)
    g3 = panel(axes[3], scene, t + 1, c, "xz", focal_t1, window=win)
    for ax, geo in ((axes[1], g1), (axes[3], g3)):
        # GT edge (green), predicted edge from the source (red dashed), competitor's edge (magenta dashed); source positions from frame t
        arrow(ax, geo, scene.gt_c[u], scene.gt_c[v], C_GT)
        if ch is not None:
            arrow(ax, geo, scene.pred_c[pu], scene.pred_c[ch], C_PRED, ls="--")
        if taker is not None:
            arrow(ax, geo, scene.pred_c[taker], scene.pred_c[pv], C_TAKER, ls="--")
            ax.plot(*geo["pos"](scene.pred_c[taker]), "x", color=C_TAKER, ms=8, mew=1.5)
        ax.plot(*geo["pos"](scene.pred_c[pu]), "x", color=C_PRED, ms=8, mew=1.5)
    fix_lims([(axes[0], g0), (axes[1], g1), (axes[2], g2), (axes[3], g3)])


def linker_title(w):
    rank = "beyond 15 µm gate" if w["rank"] == -1 else f"true partner rank {int(w['rank'])}"
    bits = [f"{w['name']}  t={int(w.t)}", f"GT moved {w.disp_um:.1f} µm", f"detections of the true pair: xy {w.true_dxy_um:.1f}, z {w.true_dz_um:+.1f} µm",
            f"score true {w.s_true:.2f}" + ("" if pd.isna(w.s_chosen) else f" vs chosen {w.s_chosen:.2f}"), rank, f"nearest other detection {w.crowd_um:.1f} µm"]
    if not pd.isna(w.taker_score):
        bits.append(f"competitor scored the true target {w.taker_score:.2f} from {w.taker_dist_um:.1f} µm away (z {w.taker_dz_um:+.1f})")
    return "; ".join(bits)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, default=Path("dist/error_examples/tables.pkl"))
    ap.add_argument("--probe", type=Path, default=Path("dist/error_examples/probe.pkl"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = pickle.load(open(args.tables, "rb"))
    probe = pickle.load(open(args.probe, "rb")) if args.probe.exists() else None
    M, W, N = categorize(tables, probe)
    text = freq_tables(M, W, N)
    print(text)
    (args.out_dir / "categories.md").write_text(text + "\n")
    M.to_csv(args.out_dir / "misses.csv", index=False); W.to_csv(args.out_dir / "linker_fn.csv", index=False)
    if args.no_figures:
        return

    # ---- detector examples: ranked categories, typical members
    det_rows = []
    order = M.mech.value_counts().index.tolist()
    order = sorted(order)  # D1..D5 already ranked by construction; keep the label order
    used = set()
    for mech in order:
        sub = M[M.mech == mech]
        stat = "snr"
        ex = typical(sub, stat, k=args.per_category, avoid_names=used)
        for e in ex:
            det_rows.append((mech, e)); used.add(e["name"])
    # one bright-cell miss as a cross-cutting example
    bright = M[M.bright & ~M.artefact & ~M.merged]
    for e in typical(bright, "snr", k=1, avoid_names=used):
        det_rows.append(("X bright cell (brightness > 8x median), no detection nearby", e))

    scenes = {}
    def scene(name):
        if name not in scenes:
            scenes[name] = Scene(name)
        return scenes[name]

    fig, axes = plt.subplots(len(det_rows), 4, figsize=(15, 3.9 * len(det_rows)), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.05]))
    for row, (mech, m) in zip(axes, det_rows):
        draw_detector_row(row, scene(m["name"]), m)
        nm = int((M.mech == mech).sum()) if not mech.startswith("X") else int(bright.shape[0])
        row[0].annotate(f"{mech}  --  {nm} of {len(M)} misses ({100 * nm / len(M):.0f}%)\n{detector_title(m)}", (0, 1.0), xycoords="axes fraction", xytext=(0, 24), textcoords="offset points", fontsize=7.5, va="bottom", ha="left", color="#f0f0f0", annotation_clip=False)
        print("detector example:", mech, "|", detector_title(m), flush=True)
    fig.suptitle("Detector misses, one row per example (category label above each row).  green o = GT cell, red + = detection; white ring = the missed cell (and its own track in t-1 / t+1);"
                 "\nred ring = detection nearest to it; yellow ring + line = the GT cell that claimed that detection.  Hollow square + dz/dy = node outside the projected slab."
                 "  xy: max over ±4 z-slices (±6.5 µm), 23 x 23 µm.  xz: max over ±8 y-voxels (±3.2 µm), ±10 slices, z at true scale.", fontsize=8, y=0.998)
    fig.subplots_adjust(hspace=0.62, wspace=0.06, top=0.965, bottom=0.005, left=0.02, right=0.99)
    fig.savefig(args.out_dir / "fig13_detector_examples.png"); plt.close(fig)

    # ---- linker examples
    link_rows = []
    used = set()
    for mech in sorted(W.mech.unique()):
        sub = W[W.mech == mech]
        k = args.per_category if not mech.startswith(("L1c", "L1d", "L4")) else 1
        for e in typical(sub, "crowd_um", k=k, avoid_names=used):
            link_rows.append((mech, e)); used.add(e["name"])
    fig, axes = plt.subplots(len(link_rows), 4, figsize=(15, 3.9 * len(link_rows)), gridspec_kw=dict(width_ratios=[1, 1, 1.05, 1.05]))
    for row, (mech, w) in zip(axes, link_rows):
        draw_linker_row(row, scene(w["name"]), w)
        nm = int((W.mech == mech).sum())
        row[0].annotate(f"{mech}  --  {nm} of {len(W)} linker FN ({100 * nm / len(W):.0f}%)\n{linker_title(w)}", (0, 1.0), xycoords="axes fraction", xytext=(0, 24), textcoords="offset points", fontsize=7.5, va="bottom", ha="left", color="#f0f0f0", annotation_clip=False)
        print("linker example:", mech, "|", linker_title(w), flush=True)
    fig.suptitle("Linker errors, one row per example.  Frame t: white ring = GT source cell, red ring = its detection (red x in t+1 panels = where the source detection was);"
                 " magenta = competing source that took the true target.\nFrame t+1: white ring = true target (GT), red ring = its detection, yellow ring = the detection the linker chose."
                 "  Green arrow = GT link, red dashed = predicted link, magenta dashed = competitor's link.  xz side views (right) expose z offsets.", fontsize=8, y=0.998)
    fig.subplots_adjust(hspace=0.62, wspace=0.06, top=0.965, bottom=0.005, left=0.02, right=0.99)
    fig.savefig(args.out_dir / "fig14_linker_examples.png"); plt.close(fig)
    json.dump(dict(detector=[(m, dict(name=e["name"], gid=int(e["gid"]), t=int(e["t"]))) for m, e in det_rows],
                   linker=[(m, dict(name=e["name"], u=int(e["u"]), v=int(e["v"]), t=int(e["t"]))) for m, e in link_rows]),
              open(args.out_dir / "examples.json", "w"), indent=1)


if __name__ == "__main__":
    main()

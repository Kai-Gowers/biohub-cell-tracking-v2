#!/usr/bin/env python3
"""Qualitative panels for the main error cases of the 0.942-port pipeline (xy and xz crops).

    python scripts/error_examples_0942.py --analysis-dir dist/error_analysis_0942 \
        --pred-dir dist/preds_val_ddp_best_blend --ilp-dir dist/preds_val_ddp_best_blend_ilp \
        --out-dir reports/figures/2026-09-17-error-analysis-0942 --n 4

Inputs are the CSVs written by error_detection_misses.py (nodes.csv) and error_stage_diff.py
(fp_edges.csv, fn_edges.csv). Categories:

  miss_offset      GT cell missed, an unmatched prediction 7-10 um away (ILP-stage node set)
  miss_dim         GT cell missed, nothing within 14 um or intensity_rel < 0.5
  fp_stay          wrong link: the source links to an unannotated cell ~2 um away while the annotated
                   child is >= 7 um away and was detected
  fn_wrong_crowded FN with both endpoints detected but linked elsewhere, source nn < 10 um

Each example: frame t xy (max over +-2 z planes) and xz (max over +-3 y rows) crops centred on the GT
node; for link errors also frame t+1 xy. Markers: GT nodes (green o), predicted nodes (red x), the GT
edge (green arrow) and the predicted edge (red arrow). Crops are 40 grid cells = 65 um wide.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cell_tracking.cache import VolumeFrames, cache_path  # noqa: E402
from cell_tracking.config import SCALE, get_train_dir  # noqa: E402
from cell_tracking.io_geff import GeffGraph, read_geff  # noqa: E402

G = 1.625  # um per grid cell (isotropic after xy decimation)
HALF = 20   # crop half-width in grid cells


def to_grid(zyx_vx: np.ndarray) -> np.ndarray:
    return np.asarray(zyx_vx, dtype=np.float64) * SCALE / G


def nodes_in_frame(g: GeffGraph, t: int) -> dict[int, np.ndarray]:
    m = g.t == t
    return {int(n): to_grid([z, y, x]) for n, z, y, x in zip(g.node_ids[m], g.z[m], g.y[m], g.x[m])}


def crop_xy(frame: np.ndarray, c: np.ndarray):
    z, y, x = np.round(c).astype(int)
    z0, z1 = max(z - 2, 0), min(z + 3, frame.shape[0])
    y0, y1 = max(y - HALF, 0), min(y + HALF, frame.shape[1])
    x0, x1 = max(x - HALF, 0), min(x + HALF, frame.shape[2])
    return frame[z0:z1, y0:y1, x0:x1].max(axis=0), (y0, x0)


def crop_xz(frame: np.ndarray, c: np.ndarray):
    z, y, x = np.round(c).astype(int)
    y0, y1 = max(y - 3, 0), min(y + 4, frame.shape[1])
    x0, x1 = max(x - HALF, 0), min(x + HALF, frame.shape[2])
    return frame[:, y0:y1, x0:x1].max(axis=1), (0, x0)


def draw(ax, img, origin, gt: dict, pred: dict, centre, plane: str, arrows=(), title="", focus=None):
    """Markers are drawn only for nodes inside the projected slab (xy: +-2 z planes; xz: +-3 y rows)."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=max(float(np.percentile(img, 99.5)), 1e-3), interpolation="nearest")
    oy, ox = origin
    a, b = (1, 2) if plane == "xy" else (0, 2)  # rows, cols of the panel in (z,y,x) indices

    def inside(p):
        if abs(p[2] - centre[2]) > HALF:
            return False
        if plane == "xy":
            return abs(p[1] - centre[1]) <= HALF and abs(p[0] - centre[0]) <= 2.5
        return abs(p[1] - centre[1]) <= 3.5

    for p in gt.values():
        if inside(p):
            ax.plot(p[b] - ox, p[a] - oy, "o", mfc="none", mec="lime", ms=9, mew=1.5)
    for p in pred.values():
        if inside(p):
            ax.plot(p[b] - ox, p[a] - oy, "x", color="red", ms=7, mew=1.5)
    if focus is not None:
        ax.plot(focus[b] - ox, focus[a] - oy, "o", mfc="none", mec="cyan", ms=16, mew=2)
    for (p, q, color) in arrows:
        ax.annotate("", xy=(q[b] - ox, q[a] - oy), xytext=(p[b] - ox, p[a] - oy),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    if plane == "xz":
        ax.set_aspect(1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, default=Path("dist/error_analysis_0942"))
    ap.add_argument("--pred-dir", type=Path, default=Path("dist/preds_val_ddp_best_blend"))
    ap.add_argument("--ilp-dir", type=Path, default=Path("dist/preds_val_ddp_best_blend_ilp"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train_dir = get_train_dir()
    cache_dir = Path("dist/cache_pack") if Path("dist/cache_pack").exists() else None
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    nodes = list(csv.DictReader(open(a.analysis_dir / "misses_ilp" / "nodes.csv")))
    fps = list(csv.DictReader(open(a.analysis_dir / "stagediff" / "fp_edges.csv")))
    fns = list(csv.DictReader(open(a.analysis_dir / "stagediff" / "fn_edges.csv")))
    cats = {
        "miss_offset": [r for r in nodes if r["detected"] == "0" and r["miss_class"] == "nearby_unmatched" and float(r["pred_nn_um"]) <= 10],
        "miss_dim": [r for r in nodes if r["detected"] == "0" and (r["miss_class"] == "none_within_14um" or float(r["intensity_rel"]) < 0.5)],
        "fp_stay": [r for r in fps if r["gt_u"] != "-1" and r["true_child_detected"] == "1" and float(r["disp_um"]) < 3 and float(r["dist_to_true_child_um"]) >= 7],
        "fn_wrong_crowded": [r for r in fns if r["cause"] == "wrong_link" and r["src_nn_um"] not in ("nan", "inf") and float(r["src_nn_um"]) < 10],
    }
    index = {}
    graphs: dict[str, tuple] = {}

    def load(vol):
        if vol not in graphs:
            gt = read_geff(train_dir / f"{vol}.geff")
            pred = read_geff(a.pred_dir / f"{vol}.geff")
            ilp = read_geff(a.ilp_dir / f"{vol}.geff")
            frames = VolumeFrames(train_dir / f"{vol}.zarr", cache_path(cache_dir, vol) if cache_dir else None)
            graphs[vol] = (gt, pred, ilp, frames)
        return graphs[vol]

    for cat, rows in cats.items():
        if not rows:
            print(f"{cat}: no rows")
            continue
        picks = rng.sample(rows, min(a.n, len(rows)))
        link = cat.startswith("f")
        ncol = 3 if link else 2
        fig, axes = plt.subplots(len(picks), ncol, figsize=(3.2 * ncol, 3.2 * len(picks)))
        axes = np.atleast_2d(axes)
        index[cat] = []
        for i, r in enumerate(picks):
            vol = r["volume"]
            gt, pred, ilp, frames = load(vol)
            gt_pos_all = {int(n): to_grid([z, y, x]) for n, z, y, x in zip(gt.node_ids, gt.z, gt.y, gt.x)}
            if not link:
                t = int(r["t"])
                centre = gt_pos_all[int(r["gt_id"])]
                p_nodes = nodes_in_frame(ilp, t)
                g_nodes = nodes_in_frame(gt, t)
                f = frames.frame(t)
                img, o = crop_xy(f, centre)
                draw(axes[i, 0], img, o, g_nodes, p_nodes, centre, "xy", focus=centre,
                     title=f"{vol} t={t}\nmissed (cyan): nearest pred {r['pred_nn_um']}um (dz {r['pred_nn_dz_um']}, dxy {r['pred_nn_dxy_um']}), I/p99={r['intensity_rel']}")
                img, o = crop_xz(f, centre)
                draw(axes[i, 1], img, o, g_nodes, p_nodes, centre, "xz", focus=centre, title="xz (max over 7 y rows)")
                index[cat].append({k: r[k] for k in ("volume", "t", "gt_id", "miss_class", "pred_nn_um", "pred_nn_dz_um", "pred_nn_dxy_um", "intensity_rel", "gt_nn_um")})
            else:
                t = int(r["t"])
                pos_pred = {int(n): to_grid([z, y, x]) for n, z, y, x in zip(pred.node_ids, pred.z, pred.y, pred.x)}
                pred_t = pred.t_of()
                if cat == "fp_stay":
                    pu, pv = int(r["pred_u"]), int(r["pred_v"])
                    src = pos_pred[pu]
                    gu = int(r["gt_u"])
                    true_children = [int(v) for u, v in gt.edges if int(u) == gu]
                    arrows_t = [(src, pos_pred[pv], "red")] + [(gt_pos_all[gu], gt_pos_all[c], "lime") for c in true_children if c in gt_pos_all]
                    title = f"{vol} t={t}->{t + 1} FP\nred: to unannotated cell {r['disp_um']}um; green: annotated child {r['dist_to_true_child_um']}um"
                    centre = src
                else:
                    gu, gv = int(r["gt_u"]), int(r["gt_v"])
                    centre = gt_pos_all[gu]
                    # predicted edges leaving the matched source / entering the matched child
                    arrows_t = [(gt_pos_all[gu], gt_pos_all[gv], "lime")]
                    for u, v in pred.edges:
                        pu_, pv_ = pos_pred[int(u)], pos_pred[int(v)]
                        if pred_t[int(u)] == t and (np.linalg.norm(pu_ - centre) < 7 / G or np.linalg.norm(pv_ - gt_pos_all[gv]) < 7 / G):
                            arrows_t.append((pu_, pv_, "red"))
                    title = f"{vol} t={t}->{t + 1} FN wrong link\nGT moved {r['disp_um']}um (dz {r['dz_um']}); source nn {r['src_nn_um']}um"
                f0, f1 = frames.frame(t), frames.frame(t + 1)
                img, o = crop_xy(f0, centre)
                draw(axes[i, 0], img, o, nodes_in_frame(gt, t), nodes_in_frame(pred, t), centre, "xy", arrows=arrows_t, title=title, focus=centre)
                img, o = crop_xy(f1, centre)
                draw(axes[i, 1], img, o, nodes_in_frame(gt, t + 1), nodes_in_frame(pred, t + 1), centre, "xy", arrows=arrows_t, title=f"t+1={t + 1} xy (same crop)")
                img, o = crop_xz(f0, centre)
                draw(axes[i, 2], img, o, nodes_in_frame(gt, t), nodes_in_frame(pred, t), centre, "xz", arrows=arrows_t, title="t xz", focus=centre)
                index[cat].append({k: r[k] for k in r if k in ("volume", "t", "gt_u", "gt_v", "pred_u", "pred_v", "disp_um", "dz_um", "src_nn_um", "dist_to_true_child_um")})
        fig.suptitle(f"{cat}  (n={len(rows)} cases; green o = annotated cell, red x = prediction, cyan = the case; arrows: green = GT edge, red = predicted)", fontsize=9)
        fig.tight_layout()
        out = a.out_dir / f"{cat}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"{cat}: {len(rows)} cases -> {out}")
    (a.out_dir / "examples_index.json").write_text(json.dumps(index, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

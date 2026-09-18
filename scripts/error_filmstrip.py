#!/usr/bin/env python3
"""Cell-centric filmstrip: one annotated track, one row per frame, crops centred on the cell.

    python scripts/error_filmstrip.py --bundle dist/error_viewer3d/6bba_3abfe10a.npz --top 3 \
        --out-dir reports/figures/2026-09-18-filmstrips

For each selected track (the ones with most error frames, or --track TID) and a window of frames around its
first error (--before / --after), draw per frame:
  left   xy crop (max over +-2 z planes around the cell), 40 um wide
  right  xz crop (max over +-3 y rows), same x range, all z
Markers: the followed annotated cell = cyan ring; other annotated cells = green rings; predictions = small
dots (grey no counted link, green correct link leaves it, red false link leaves it); the prediction matched
to the followed cell = cyan dot joined to the ring; predicted links leaving any node in the crop = arrows to
the node's position in the next frame (green correct, red false, grey uncounted); the annotated link of the
followed cell = dashed cyan arrow. Row label = frame and the cell's outcome that frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STATUS_COLORS = {"link_ok": "#56C271", "link_wrong": "#F2A93B", "dropped": "#D86CD1", "weak": "#EF6363", "none": "#8C96A1"}
STATUS_LABEL = {"link_ok": "linked ok", "link_wrong": "WRONG LINK", "dropped": "DROPPED BY ILP (peak existed)",
                "weak": "MISSED (weak peak)", "none": "MISSED (no peak)"}
PRED_COLORS = ["#B0B8C4", "#56C271", "#FF5C5C"]
HALF_UM = 20.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--track", type=int, action="append", default=None)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--before", type=int, default=3)
    ap.add_argument("--after", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    b = np.load(a.bundle, allow_pickle=False)
    meta = json.loads(str(b["meta"]))
    T, Z, Y, X = meta["shape"]
    sz, sy, sx = meta["scale_um"]
    ds = meta.get("xy_downsample", 1)
    status = meta["status"]
    frames = b["frames"]
    gt, gstat, gtr, gmatch = b["gt"], np.array(status)[b["gt_status"]], b["gt_track"], b["gt_match"]
    pred, pstat = b["pred"], b["pred_status"]
    gl, pl, pc = b["gt_links"], b["pred_links"], b["pred_link_cls"]
    tracks = b["tracks"]
    tracks = tracks[np.argsort(-tracks["errors"], kind="stable")]
    tids = a.track or [int(tr["tid"]) for tr in tracks if tr["n"] >= 2][: a.top]

    # index by frame
    gt_by_t = {t: np.nonzero(gt[:, 0] == t)[0] for t in range(T)}
    pred_by_t = {t: np.nonzero(pred[:, 0] == t)[0] for t in range(T)}
    pl_by_t = {t: np.nonzero(pl[:, 0, 0] == t)[0] for t in range(T)}
    gl_by_t = {t: np.nonzero(gl[:, 0, 0] == t)[0] for t in range(T)}

    for tid in tids:
        rows_t = np.nonzero(gtr == tid)[0]
        tr = tracks[tracks["tid"] == tid][0]
        t_err = int(tr["first_error"]) if tr["first_error"] >= 0 else int(gt[rows_t][:, 0].min())
        ts = [t for t in range(t_err - a.before, t_err + a.after + 1) if 0 <= t < T and (gt[rows_t][:, 0] == t).any()]
        if not ts:
            continue
        fig, axes = plt.subplots(len(ts), 2, figsize=(9.2, 3.3 * len(ts)), gridspec_kw={"width_ratios": [1, 1]})
        axes = np.atleast_2d(axes)
        for r, t in enumerate(ts):
            i = rows_t[gt[rows_t][:, 0] == t][0]
            cz, cy, cx = gt[i][1:]                         # raw voxels
            cyum, cxum, czum = cy * sy, cx * sx, cz * sz
            # crop bounds in raw voxels
            hy, hx = HALF_UM / sy, HALF_UM / sx
            y0, y1 = int(max(0, cy - hy)), int(min(Y, cy + hy)); x0, x1 = int(max(0, cx - hx)), int(min(X, cx + hx))
            z0, z1 = int(max(0, cz - 2)), int(min(Z, cz + 3))
            f = frames[t]
            xy = f[z0:z1, y0 // ds:y1 // ds, x0 // ds:x1 // ds].max(axis=0)
            yy0, yy1 = int(max(0, cy - 3)), int(min(Y, cy + 4))
            xz = f[:, yy0 // ds:yy1 // ds, x0 // ds:x1 // ds].max(axis=1)
            ax_xy, ax_xz = axes[r]
            ax_xy.imshow(xy, cmap="gray", extent=[x0 * sx, x1 * sx, y1 * sy, y0 * sy], vmin=0, vmax=max(xy.max(), 1))
            ax_xz.imshow(xz, cmap="gray", extent=[x0 * sx, x1 * sx, Z * sz, 0], vmin=0, vmax=max(xz.max(), 1), aspect="auto")

            def in_xy(p):  # p = [t,z,y,x] raw
                return x0 <= p[3] < x1 and y0 <= p[2] < y1 and abs(p[1] - cz) <= 2.5
            def in_xz(p):
                return x0 <= p[3] < x1 and abs(p[2] - cy) <= 3.5
            # predictions
            for j in pred_by_t[t]:
                p = pred[j]; col = PRED_COLORS[pstat[j]]
                if in_xy(p): ax_xy.plot(p[3] * sx, p[2] * sy, "o", ms=4, color=col, mec="black", mew=0.3)
                if in_xz(p): ax_xz.plot(p[3] * sx, p[1] * sz, "o", ms=4, color=col, mec="black", mew=0.3)
            # predicted links leaving nodes in the crop
            for k in pl_by_t[t]:
                s_, d_ = pl[k]; col = PRED_COLORS[pc[k]]; lw = 1.6 if pc[k] == 2 else 1.0
                if in_xy(s_): ax_xy.annotate("", xy=((s_[3] + d_[3]) * sx, (s_[2] + d_[2]) * sy), xytext=(s_[3] * sx, s_[2] * sy), arrowprops=dict(arrowstyle="->", color=col, lw=lw))
                if in_xz(s_): ax_xz.annotate("", xy=((s_[3] + d_[3]) * sx, (s_[1] + d_[1]) * sz), xytext=(s_[3] * sx, s_[1] * sz), arrowprops=dict(arrowstyle="->", color=col, lw=lw))
            # other annotated cells
            for j in gt_by_t[t]:
                if j == i: continue
                p = gt[j]
                if in_xy(p): ax_xy.plot(p[3] * sx, p[2] * sy, "o", ms=9, mfc="none", mec="#56C271", mew=1.2)
                if in_xz(p): ax_xz.plot(p[3] * sx, p[1] * sz, "o", ms=9, mfc="none", mec="#56C271", mew=1.2)
            # the followed cell, its annotated link, and its matched prediction
            for ax, yv in ((ax_xy, cyum), (ax_xz, czum)):
                ax.plot(cxum, yv, "o", ms=13, mfc="none", mec="#4FD1E0", mew=2.2)
            for k in gl_by_t[t]:
                s_, d_ = gl[k]
                if abs(s_[2] - cy) < 0.5 and abs(s_[3] - cx) < 0.5 and abs(s_[1] - cz) < 0.5:
                    ax_xy.annotate("", xy=((s_[3] + d_[3]) * sx, (s_[2] + d_[2]) * sy), xytext=(cxum, cyum), arrowprops=dict(arrowstyle="->", color="#4FD1E0", lw=1.6, ls="--"))
                    ax_xz.annotate("", xy=((s_[3] + d_[3]) * sx, (s_[1] + d_[1]) * sz), xytext=(cxum, czum), arrowprops=dict(arrowstyle="->", color="#4FD1E0", lw=1.6, ls="--"))
            if gmatch[i] >= 0:
                q = pred[gmatch[i]]
                ax_xy.plot([cxum, q[3] * sx], [cyum, q[2] * sy], "-", color="#4FD1E0", lw=1.2); ax_xy.plot(q[3] * sx, q[2] * sy, "o", ms=5, color="#4FD1E0", mec="black", mew=0.4)
                ax_xz.plot([cxum, q[3] * sx], [czum, q[1] * sz], "-", color="#4FD1E0", lw=1.2); ax_xz.plot(q[3] * sx, q[1] * sz, "o", ms=5, color="#4FD1E0", mec="black", mew=0.4)
            st = gstat[i]
            off = f", matched prediction {np.linalg.norm((pred[gmatch[i]][1:] - gt[i][1:]) * np.array([sz, sy, sx])):.1f} um off" if gmatch[i] >= 0 else ""
            ax_xy.set_title(f"t = {t}   {STATUS_LABEL[st]}\n{off.lstrip(', ') or ' '}", fontsize=9.5, color=STATUS_COLORS[st], loc="left",
                            fontweight="bold" if st != "link_ok" else "normal")
            ax_xz.set_title(f"xz  (z = {czum:.0f} um)\n ", fontsize=9, loc="left", color="#666")
            for ax in (ax_xy, ax_xz):
                ax.set_xticks([]); ax.set_yticks([])
            ax_xy.set_ylabel("y", fontsize=8); ax_xz.set_ylabel("z", fontsize=8)
        fig.suptitle(f"{meta['name']}  annotated track #{tid}  ({tr['errors']} error frames of {tr['n']}, first at t = {t_err})\n"
                     "cyan ring = the cell   dashed cyan arrow = where the annotation says it goes   cyan dot = prediction matched to it\n"
                     "green rings = other annotated cells   dots = predictions (grey no counted link, green correct, red false)   arrows = predicted links. Crops 40 um.",
                     fontsize=8.5, ha="center")
        fig.tight_layout(rect=(0, 0, 1, 1 - 1.3 / (3.05 * len(ts))))
        out = a.out_dir / f"{meta['name']}_track{tid}.png"
        fig.savefig(out, dpi=110); plt.close(fig)
        print(f"track #{tid}: frames {ts[0]}-{ts[-1]} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

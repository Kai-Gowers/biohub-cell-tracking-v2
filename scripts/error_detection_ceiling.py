#!/usr/bin/env python3
"""Is the evidence for a missed cell present in the detection map? (Ultrack-style hypothesis ceiling.)

    python scripts/error_detection_ceiling.py --det-dir <dump-det-dir> --nodes dist/error_analysis_0942/misses_ilp/nodes.csv \
        --out-dir dist/error_analysis_0942/ceiling

Input: the per-frame detection probability maps written by `predict.py --dump-det-dir` (float16, 1.625 um
grid, exactly what the peak picker sees) and the per-GT-cell table from error_detection_misses.py.

For every annotated cell we look at the map within the metric's 7 um matching radius (a ball of 4.3 grid
cells) and record: the probability at the cell (max over 3x3x3), the max probability within 7 um, and the
local maxima (3x3x3 max-pool equality, the same NMS the pipeline uses) within 7 um: how many, the best one's
probability and distance. A missed cell whose best local maximum within 7 um has probability p could have
been emitted as a candidate at threshold p; the pipeline emits only p > 0.965.

Also counts, per frame, how many extra local maxima appear when the threshold is lowered (candidate
inflation), so the recall ceiling can be weighed against the extra hypotheses the ILP would have to reject.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

G = 1.625
R_CELLS = 7.0 / G
THRESHOLDS = [0.965, 0.9, 0.8, 0.5, 0.3, 0.1]


def ball_offsets(r: float) -> np.ndarray:
    n = int(np.ceil(r))
    zz, yy, xx = np.mgrid[-n:n + 1, -n:n + 1, -n:n + 1]
    m = (zz ** 2 + yy ** 2 + xx ** 2) <= r * r
    return np.stack([zz[m], yy[m], xx[m]], axis=1)


OFFS = ball_offsets(R_CELLS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dir", type=Path, required=True)
    ap.add_argument("--nodes", type=Path, default=Path("dist/error_analysis_0942/misses_ilp/nodes.csv"))
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.nodes)))
    by_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_frame[(r["volume"], int(r["t"]))].append(r)

    out_rows = []
    inflation = defaultdict(list)  # threshold -> extra candidates per frame relative to 0.965
    frames_done = 0
    for (vol, t), rs in sorted(by_frame.items()):
        f = a.det_dir / vol / f"t{t:03d}.npy"
        if not f.exists():
            continue
        p = np.squeeze(np.load(f).astype(np.float32))  # (Z, Y, X) on the grid (dump carries a channel axis)
        mx = maximum_filter(p, size=3, mode="nearest")
        is_max = p == mx
        base = int(((p > 0.965) & is_max).sum())
        for thr in THRESHOLDS:
            inflation[thr].append(int(((p > thr) & is_max).sum()) - base)
        frames_done += 1
        Z, Y, X = p.shape
        for r in rs:
            c = np.array([float(r["z_um"]), float(r["y_um"]), float(r["x_um"])]) / G
            ci = np.round(c).astype(int)
            pts = ci + OFFS
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < Z) & (pts[:, 1] >= 0) & (pts[:, 1] < Y) & (pts[:, 2] >= 0) & (pts[:, 2] < X)
            pts = pts[ok]
            vals = p[pts[:, 0], pts[:, 1], pts[:, 2]]
            lm = is_max[pts[:, 0], pts[:, 1], pts[:, 2]]
            dist = np.linalg.norm((pts - c) * G, axis=1)
            near = dist <= 1.7  # ~1 grid cell
            p_at = float(vals[near].max()) if near.any() else float("nan")
            p_max7 = float(vals.max()) if len(vals) else float("nan")
            if lm.any():
                j = np.argmax(np.where(lm, vals, -1))
                best_lm_p, best_lm_d = float(vals[j]), float(dist[j])
                n_lm = int(lm.sum())
                lm_above = {thr: int(((vals > thr) & lm).sum()) for thr in THRESHOLDS}
            else:
                best_lm_p, best_lm_d, n_lm = float("nan"), float("nan"), 0
                lm_above = {thr: 0 for thr in THRESHOLDS}
            out_rows.append({
                "volume": vol, "embryo": r["embryo"], "t": t, "gt_id": r["gt_id"], "detected": r["detected"], "miss_class": r["miss_class"],
                "p_at": round(p_at, 4), "p_max7": round(p_max7, 4), "n_localmax7": n_lm,
                "best_localmax_p": round(best_lm_p, 4), "best_localmax_dist_um": round(best_lm_d, 2),
                **{f"lm_gt_{thr}": lm_above[thr] for thr in THRESHOLDS},
            })

    with open(a.out_dir / "ceiling_nodes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    def summarise(rs: list[dict]) -> dict:
        det = np.array([r["detected"] == "1" for r in rs])
        best = np.array([r["best_localmax_p"] for r in rs], dtype=float)
        pmax = np.array([r["p_max7"] for r in rs], dtype=float)
        s = {"n": len(rs), "n_missed": int((~det).sum())}
        for label, m in [("detected", det), ("missed", ~det)]:
            b, pm = best[m], pmax[m]
            s[label] = {
                "p_max7_median": float(np.nanmedian(pm)) if m.any() else None,
                "best_localmax_p_median": float(np.nanmedian(b)) if m.any() else None,
                "no_localmax_within_7um": int(np.isnan(b).sum()),
                **{f"frac_localmax_gt_{thr}": float(np.mean(np.nan_to_num(b, nan=-1) > thr)) for thr in THRESHOLDS},
            }
        return s

    summary = {"frames": frames_done, "all": summarise(out_rows)}
    for emb in sorted({r["embryo"] for r in out_rows}):
        summary[emb] = summarise([r for r in out_rows if r["embryo"] == emb])
    for cls in sorted({r["miss_class"] for r in out_rows if r["detected"] == "0"}):
        summary[f"miss_class:{cls}"] = summarise([r for r in out_rows if r["miss_class"] == cls])
    summary["candidate_inflation_per_frame"] = {
        str(thr): {"median_extra": float(np.median(v)), "mean_extra": float(np.mean(v))} for thr, v in inflation.items()
    }
    base_per_frame = None
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"frames analysed: {frames_done}")
    for key in ["all", "44b6", "6bba"] + [k for k in summary if k.startswith("miss_class:")]:
        if key not in summary:
            continue
        s = summary[key]
        print(f"\n=== {key}: n={s['n']} missed={s['n_missed']}")
        for label in ("detected", "missed"):
            d = s[label]
            if d["p_max7_median"] is None:
                continue
            fr = " ".join(f">{thr}:{100 * d[f'frac_localmax_gt_{thr}']:.0f}%" for thr in THRESHOLDS)
            print(f"  {label:9s} max p within 7um median {d['p_max7_median']:.3f} | best local max p median {d['best_localmax_p_median']:.3f} |"
                  f" no local max within 7um: {d['no_localmax_within_7um']} | has a local max above: {fr}")
    print("\ncandidate inflation per frame (extra local maxima vs threshold 0.965):")
    for thr, v in summary["candidate_inflation_per_frame"].items():
        print(f"  threshold {thr}: median +{v['median_extra']:.0f}, mean +{v['mean_extra']:.0f} per frame")
    print(f"\nwrote {a.out_dir}/ceiling_nodes.csv, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Profile every annotated (GT) cell of the held-out volumes: detected or missed, and if missed, why.

    python scripts/error_detection_misses.py --geff-dir dist/preds_val_<tag> --split dist/heldout_split.json \
        --out-dir dist/error_analysis_0942/<tag>

Detection here means the metric's own 7 um per-frame bipartite matching (`metric.match_nodes_per_frame`),
so "missed" == a GT node that no counted edge can ever touch. For every GT node we record

  z_um, t, boundary_um (distance to the nearest volume face), gt_nn_um (nearest annotated cell in the
  frame; sparse, so a lower bound on crowding), pred_nn_um (nearest predicted node in the frame),
  intensity (normalised frame value, max over a 3x3x3 grid neighbourhood at the GT location; the same
  normalisation the model sees), frame_p99 (99th percentile of that frame) and intensity_rel = intensity / p99.

Missed nodes are classed by the nearest prediction:
  matched_elsewhere   a prediction within 7 um exists but the assignment gave it to another GT node
  nearby_offset       nearest prediction 7-14 um away and matched to another GT (peak merged with a neighbour)
  nearby_unmatched    nearest prediction 7-14 um away and unmatched (displaced / split detection)
  none_within_14um    nothing within 14 um -- a genuine non-detection

Writes nodes.csv (one row per GT node) and summary.json, and prints the miss rate by z, intensity, crowding,
boundary distance and frame, per embryo.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cell_tracking.cache import VolumeFrames, cache_path  # noqa: E402
from cell_tracking.config import SCALE, get_cache_dir, get_train_dir  # noqa: E402
from cell_tracking.io_geff import embryo_of, read_geff  # noqa: E402
from cell_tracking.io_zarr import read_array_meta  # noqa: E402
from cell_tracking.metric import match_nodes_per_frame  # noqa: E402

GRID = np.array([1.0, 4.0, 4.0])  # raw voxel -> model grid (z, y, x)


def profile_volume(name: str, train_dir: Path, geff_dir: Path, cache_dir: Path | None) -> list[dict]:
    gt = read_geff(train_dir / f"{name}.geff")
    pred = read_geff(geff_dir / f"{name}.geff")
    gt_by_t, pred_by_t = gt.nodes_by_t(), pred.nodes_by_t()
    pred_to_gt = match_nodes_per_frame(gt_by_t, pred_by_t)
    matched_gt = set(pred_to_gt.values())
    shape, _ = read_array_meta(train_dir / f"{name}.zarr")  # (T, Z, Y, X)
    extent_um = np.array(shape[1:], dtype=np.float64) * SCALE
    frames = VolumeFrames(train_dir / f"{name}.zarr", cache_path(cache_dir, name) if cache_dir else None)
    n_t = shape[0]

    rows: list[dict] = []
    for t in sorted(gt_by_t):
        gl = gt_by_t[t]
        gt_ids = [nid for nid, _ in gl]
        gt_vx = np.array([c for _, c in gl], dtype=np.float64)
        gt_um = gt_vx * SCALE
        pl = pred_by_t.get(t, [])
        pred_ids = [nid for nid, _ in pl]
        pred_um = np.array([c for _, c in pl], dtype=np.float64) * SCALE if pl else np.zeros((0, 3))
        gt_tree = cKDTree(gt_um) if len(gt_um) > 1 else None
        pred_tree = cKDTree(pred_um) if len(pred_um) else None
        frame = frames.frame(t)  # normalised, decimated (Z, Y/4, X/4)
        p99 = float(np.percentile(frame, 99))
        for i, gid in enumerate(gt_ids):
            z, y, x = gt_um[i]
            boundary = float(min(z, extent_um[0] - z, y, extent_um[1] - y, x, extent_um[2] - x))
            gt_nn = float(gt_tree.query(gt_um[i], k=2)[0][1]) if gt_tree is not None else float("inf")
            if pred_tree is not None:
                d, j = pred_tree.query(gt_um[i], k=1)
                pred_nn, pred_nn_id = float(d), int(pred_ids[int(j)])
                off = pred_um[int(j)] - gt_um[i]
                pred_nn_dz, pred_nn_dxy = float(abs(off[0])), float(np.hypot(off[1], off[2]))
            else:
                pred_nn, pred_nn_id = float("inf"), -1
                pred_nn_dz, pred_nn_dxy = float("nan"), float("nan")
            gz, gy, gx = np.round(gt_vx[i] / GRID).astype(int)
            z0, z1 = max(gz - 1, 0), min(gz + 2, frame.shape[0])
            y0, y1 = max(gy - 1, 0), min(gy + 2, frame.shape[1])
            x0, x1 = max(gx - 1, 0), min(gx + 2, frame.shape[2])
            patch = frame[z0:z1, y0:y1, x0:x1]
            inten = float(patch.max()) if patch.size else float("nan")
            detected = gid in matched_gt
            if detected:
                cls = "detected"
            elif pred_nn <= 7.0:
                cls = "matched_elsewhere"
            elif pred_nn <= 14.0:
                cls = "nearby_offset" if pred_nn_id in pred_to_gt else "nearby_unmatched"
            else:
                cls = "none_within_14um"
            rows.append({
                "volume": name, "embryo": embryo_of(name), "gt_id": int(gid), "t": int(t), "t_frac": t / max(n_t - 1, 1),
                "z_um": round(float(z), 2), "y_um": round(float(y), 2), "x_um": round(float(x), 2),
                "boundary_um": round(boundary, 2), "gt_nn_um": round(gt_nn, 2), "pred_nn_um": round(pred_nn, 2),
                "pred_nn_dz_um": round(pred_nn_dz, 2), "pred_nn_dxy_um": round(pred_nn_dxy, 2),
                "intensity": round(inten, 4), "frame_p99": round(p99, 4),
                "intensity_rel": round(inten / p99, 3) if p99 > 0 else float("nan"),
                "detected": int(detected), "miss_class": cls,
            })
    return rows


def _bins(values: np.ndarray, edges: list[float], labels: list[str]) -> np.ndarray:
    idx = np.digitize(values, edges)  # 0..len(edges)
    return np.array(labels)[idx]


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    det = np.array([r["detected"] for r in rows], dtype=bool)
    out["n_gt"] = int(len(rows))
    out["n_missed"] = int((~det).sum())
    out["recall"] = float(det.mean()) if len(rows) else float("nan")
    classes = defaultdict(int)
    for r in rows:
        if not r["detected"]:
            classes[r["miss_class"]] += 1
    out["miss_classes"] = dict(classes)

    def table(key: str, edges: list[float], labels: list[str], transform=lambda v: v) -> dict:
        vals = np.array([transform(r[key]) for r in rows], dtype=np.float64)
        b = _bins(vals, edges, labels)
        tab = {}
        for lab in labels:
            m = b == lab
            if m.sum():
                tab[lab] = {"n": int(m.sum()), "missed": int((~det[m]).sum()), "miss_rate": round(float((~det[m]).mean()), 4)}
        return tab

    out["by_z_um"] = table("z_um", [20, 40, 60, 80], ["<20", "20-40", "40-60", "60-80", ">=80"])
    out["by_intensity_rel"] = table("intensity_rel", [0.25, 0.5, 0.75, 1.0], ["<0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", ">=1.0"])
    out["by_gt_nn_um"] = table("gt_nn_um", [7, 10, 14, 20], ["<7", "7-10", "10-14", "14-20", ">=20"])
    out["by_boundary_um"] = table("boundary_um", [3.5, 7, 14], ["<3.5", "3.5-7", "7-14", ">=14"])
    out["by_t_frac"] = table("t_frac", [0.01, 0.25, 0.5, 0.75, 0.99], ["t=0", "0-25%", "25-50%", "50-75%", "75-99%", "t=last"])
    return out


def print_summary(title: str, s: dict) -> None:
    print(f"\n=== {title}: {s['n_gt']} GT nodes, {s['n_missed']} missed, recall {100 * s['recall']:.1f}% ===")
    print("missed by nearest prediction:")
    for k in ["none_within_14um", "nearby_offset", "nearby_unmatched", "matched_elsewhere"]:
        v = s["miss_classes"].get(k, 0)
        print(f"  {k:20s} {v:5d}  {100 * v / max(s['n_missed'], 1):5.1f}%")
    for key, label in [("by_z_um", "depth z (um)"), ("by_intensity_rel", "intensity / frame p99"),
                       ("by_gt_nn_um", "nearest annotated neighbour (um)"), ("by_boundary_um", "distance to volume face (um)"),
                       ("by_t_frac", "frame position")]:
        print(f"miss rate by {label}:")
        for b, v in s[key].items():
            print(f"  {b:10s} n={v['n']:6d} missed={v['missed']:5d}  {100 * v['miss_rate']:5.1f}%")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--geff-dir", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    train_dir = a.train_dir or get_train_dir()
    cache_dir = a.cache_dir or Path("dist/cache_pack")
    if not cache_dir.exists():
        cache_dir = None
    names = json.loads(a.split.read_text())["datasets"]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, n in enumerate(names):
        r = profile_volume(n, train_dir, a.geff_dir, cache_dir)
        rows.extend(r)
        missed = sum(1 for x in r if not x["detected"])
        print(f"[{i + 1}/{len(names)}] {n}: {len(r)} GT nodes, {missed} missed", flush=True)

    with open(a.out_dir / "nodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"all": summarize(rows)}
    for emb in sorted({r["embryo"] for r in rows}):
        summary[emb] = summarize([r for r in rows if r["embryo"] == emb])
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print_summary("ALL held-out", summary["all"])
    for emb in sorted(k for k in summary if k != "all"):
        print_summary(f"embryo {emb}", summary[emb])
    print(f"\nwrote {a.out_dir / 'nodes.csv'} and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

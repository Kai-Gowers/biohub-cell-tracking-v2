#!/usr/bin/env python3
"""Bundle one volume for the napari 3D track-error viewer (scripts/error_viewer_napari.py).

    python scripts/error_viewer3d_build.py --volume 6bba_3abfe10a \
        --viewer-json dist/error_viewer/data/6bba_3abfe10a.json --out dist/error_viewer3d/6bba_3abfe10a.npz

Reads the per-frame analysis JSON written by scripts/error_viewer_build.py (annotated cells with their
outcome, predicted nodes and links classified by the metric, ILP-dropped candidates) and the raw zarr
frames, and writes one compressed .npz:

  frames        uint8 (T, Z, Y, X)   raw frames, per-video quantile normalised, gamma 0.5, clipped at the 99.7th pct
  gt            float32 (N, 4)       [t, z, y, x] of every annotated cell (raw voxel units)
  gt_status     int8 (N,)            0 link_ok, 1 link_wrong, 2 dropped, 3 weak, 4 none
  gt_track      int32 (N,)           annotated track id
  gt_id         int64 (N,)
  gt_links      float32 (M, 2, 4)    annotated links as vectors [[t,z,y,x],[1,dz,dy,dx]]
  pred          float32 (P, 4)       predicted nodes (final output)
  pred_status   int8 (P,)            0 no counted link leaving, 1 correct link leaves it, 2 false link leaves it
  pred_track    int32 (P,)           predicted track id (chains through single-child links)
  pred_links    float32 (L, 2, 4)    predicted links as vectors, with pred_link_cls int8 (L,): 0 ignored, 1 TP, 2 FP
  dropped       float32 (D, 4)       above-threshold detection candidates the ILP dropped, with dropped_p (D,)
  tracks        structured (K,)      tid, start, end, n, errors, first_error (-1 if none)
  meta          json string          name, shape, scale, status names
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cell_tracking.io_zarr import read_array_meta, read_attrs, read_volume  # noqa: E402
from cell_tracking.preprocess import pack_normalize, video_quantiles  # noqa: E402

STATUS = ["link_ok", "link_wrong", "dropped", "weak", "none"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--train-dir", type=Path, default=Path("data/biohub-cell-tracking-during-development/train"))
    ap.add_argument("--viewer-json", type=Path, default=None, help="default dist/error_viewer/data/<volume>.json")
    ap.add_argument("--out", type=Path, default=None, help="default dist/error_viewer3d/<volume>.npz")
    ap.add_argument("--xy-downsample", type=int, default=1, help="2 halves the bundle (frames only; coordinates stay raw)")
    a = ap.parse_args()
    vj = a.viewer_json or Path("dist/error_viewer/data") / f"{a.volume}.json"
    out = a.out or Path("dist/error_viewer3d") / f"{a.volume}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(vj.read_text())

    zarr_path = a.train_dir / f"{a.volume}.zarr"
    shape, dtype = read_array_meta(zarr_path)
    T, Z, Y, X = shape
    q_lo, q_hi = video_quantiles(read_attrs(zarr_path))
    ds = max(1, a.xy_downsample)
    frames = np.zeros((T, Z, Y // ds, X // ds), dtype=np.uint8)
    for t in range(T):
        raw = read_volume(zarr_path, t, shape, dtype)
        if ds > 1:
            raw = raw[:, ::ds, ::ds]
        v = pack_normalize(raw, q_lo, q_hi)
        hi = max(float(np.percentile(v, 99.7)), 1e-3)
        frames[t] = (np.sqrt(np.clip(v / hi, 0, 1)) * 255).astype(np.uint8)
        if t % 20 == 0:
            print(f"  frame {t}/{T}", flush=True)

    gt, gt_status, gt_track, gt_id = [], [], [], []
    pos = {}
    pred, pred_id = [], []
    dropped, dropped_p = [], []
    for t, f in enumerate(d["frames"]):
        for g in f["gt"]:
            gt.append([t, g[1], g[2], g[3]]); gt_status.append(STATUS.index(g[5])); gt_track.append(g[4]); gt_id.append(g[0])
            pos[g[0]] = np.array([t, g[1], g[2], g[3]], dtype=np.float32)
        for p in f["pred"]:
            pred.append([t, p[1], p[2], p[3]]); pred_id.append(p[0])
        for c in f["dropped"]:
            dropped.append([t, c[0], c[1], c[2]]); dropped_p.append(c[3])
    gt = np.array(gt, dtype=np.float32); pred = np.array(pred, dtype=np.float32)
    ppos = {pid: pred[i] for i, pid in enumerate(pred_id)}
    pidx = {pid: i for i, pid in enumerate(pred_id)}

    gt_links = []
    for u, v in d["gt_edges"]:
        if u in pos and v in pos:
            gt_links.append([pos[u], pos[v] - pos[u]])
    pred_links, pred_link_cls = [], []
    pred_status = np.zeros(len(pred), dtype=np.int8)
    children = defaultdict(list)
    for f in d["frames"]:
        for u, v, cls in f["pedges"]:
            if u in ppos and v in ppos:
                pred_links.append([ppos[u], ppos[v] - ppos[u]]); pred_link_cls.append(cls)
                children[u].append(v)
                if cls and pred_status[pidx[u]] < cls:
                    pred_status[pidx[u]] = cls
    # predicted track ids: chain single-child links, new id at divisions
    has_parent = {v for kids in children.values() for v in kids}
    pred_track = np.full(len(pred), -1, dtype=np.int32)
    tid = 0
    for pid in pred_id:
        if pid in has_parent:
            continue
        stack = [(pid, None)]
        while stack:
            n, cur = stack.pop()
            if cur is None:
                cur = tid; tid += 1
            pred_track[pidx[n]] = cur
            kids = children.get(n, [])
            if len(kids) == 1:
                stack.append((kids[0], cur))
            else:
                stack.extend((k, None) for k in kids)

    tracks = np.array([(tr["tid"], tr["start"], tr["end"], tr["n"], tr["errors"], -1 if tr["first_error"] is None else tr["first_error"])
                       for tr in d["tracks"]],
                      dtype=[("tid", "i4"), ("start", "i4"), ("end", "i4"), ("n", "i4"), ("errors", "i4"), ("first_error", "i4")])
    meta = {"name": a.volume, "shape": [T, Z, Y, X], "scale_um": [1.625, 0.40625, 0.40625], "xy_downsample": ds,
            "status": STATUS, "summary": d["summary"]}
    np.savez_compressed(
        out, frames=frames, gt=gt, gt_status=np.array(gt_status, dtype=np.int8), gt_track=np.array(gt_track, dtype=np.int32),
        gt_id=np.array(gt_id, dtype=np.int64), gt_links=np.array(gt_links, dtype=np.float32).reshape(-1, 2, 4),
        pred=pred, pred_status=pred_status, pred_track=pred_track,
        pred_links=np.array(pred_links, dtype=np.float32).reshape(-1, 2, 4), pred_link_cls=np.array(pred_link_cls, dtype=np.int8),
        dropped=np.array(dropped, dtype=np.float32).reshape(-1, 4), dropped_p=np.array(dropped_p, dtype=np.float32),
        tracks=tracks, meta=json.dumps(meta),
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB): frames {frames.shape}, gt {len(gt)}, pred {len(pred)}, "
          f"gt links {len(gt_links)}, pred links {len(pred_links)}, dropped {len(dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

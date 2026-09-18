#!/usr/bin/env python3
"""What the post-processing stage does to the counted edges: compare an ILP-stage dump with the full output.

    python scripts/error_stage_diff.py --ilp-dir dist/preds_val_<tag>_ilp --full-dir dist/preds_val_<tag> \
        --split dist/heldout_split.json --out-dir dist/error_analysis_0942/<tag>_stagediff

Both graphs are matched to the ground truth independently with the metric's 7 um per-frame matching, and
every *counted* edge (both endpoints matched to GT) is keyed by its GT node pair, so node ids and the
line-fit coordinate changes do not matter. For each held-out volume:

  kept      counted edge present at both stages         (TP or FP in both)
  added     counted edge only in the full output        (post-processing created it: TP = fixed, FP = new error)
  removed   counted edge only at the ILP stage          (post-processing deleted it: TP = damage, FP = cleaned)
  nodes lost / gained: GT nodes matched at one stage but not the other (short-track filter removes nodes;
                       gap closing adds synthetic ones)

Also lists the linker errors of the FULL output with their geometry (displacement, dz, crowding) so the
"main error cases" can be picked out: fp_edges.csv and fn_edges.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cell_tracking.config import SCALE, get_train_dir  # noqa: E402
from cell_tracking.io_geff import GeffGraph, embryo_of, read_geff  # noqa: E402
from cell_tracking.metric import match_nodes_per_frame  # noqa: E402


def counted_edges(gt: GeffGraph, pred: GeffGraph):
    """(pred_to_gt, {key: status}, gt_edges) for every *counted* predicted edge, per `metric.score_edges`.

    TP: both endpoints matched and the GT pair is a GT edge; key = ('tp', gu, gv).
    FP: not TP, and a matched endpoint's GT connectivity is contradicted (source has a GT child, or target
    has a GT parent); an edge touching only unmatched nodes is ignored by the metric and by us. Keyed by
    the contradicted GT endpoints, so node ids / smoothed coordinates do not matter across stages:
    key = ('fp', gu_or_None, gv_or_None).
    """
    pred_to_gt = match_nodes_per_frame(gt.nodes_by_t(), pred.nodes_by_t())
    gt_edges = {(int(u), int(v)) for u, v in gt.edges}
    has_child = {u for u, _ in gt_edges}
    has_parent = {v for _, v in gt_edges}
    counted: dict[tuple, str] = {}
    for u, v in pred.edges:
        gu, gv = pred_to_gt.get(int(u)), pred_to_gt.get(int(v))
        if gu is not None and gv is not None and (gu, gv) in gt_edges:
            counted[("tp", gu, gv)] = "tp"
            continue
        src_bad = gu is not None and gu in has_child
        dst_bad = gv is not None and gv in has_parent
        if src_bad or dst_bad:
            counted[("fp", gu if src_bad else None, gv if dst_bad else None)] = "fp"
    return pred_to_gt, counted, gt_edges


def um(g: GeffGraph) -> dict[int, np.ndarray]:
    return {int(n): np.array([z, y, x], dtype=np.float64) * SCALE for n, z, y, x in zip(g.node_ids, g.z, g.y, g.x)}


def nn_by_frame(g: GeffGraph, pos: dict[int, np.ndarray]) -> dict[int, float]:
    out: dict[int, float] = {}
    for t, nodes in g.nodes_by_t().items():
        ids = [nid for nid, _ in nodes]
        if len(ids) < 2:
            out.update({i: float("inf") for i in ids})
            continue
        P = np.array([pos[i] for i in ids])
        d, _ = cKDTree(P).query(P, k=2)
        out.update({i: float(x) for i, x in zip(ids, d[:, 1])})
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ilp-dir", type=Path, required=True)
    p.add_argument("--full-dir", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    train_dir = a.train_dir or get_train_dir()
    names = json.loads(a.split.read_text())["datasets"]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    per_volume = []
    fp_rows, fn_rows = [], []
    tot = Counter()
    for n in names:
        gt = read_geff(train_dir / f"{n}.geff")
        ilp = read_geff(a.ilp_dir / f"{n}.geff")
        full = read_geff(a.full_dir / f"{n}.geff")
        p2g_i, c_i, gt_edges = counted_edges(gt, ilp)
        p2g_f, c_f, _ = counted_edges(gt, full)
        keys_i, keys_f = set(c_i), set(c_f)
        row = {"volume": n, "embryo": embryo_of(n),
               "gt_edges": len(gt_edges),
               "ilp_tp": sum(s == "tp" for s in c_i.values()), "ilp_fp": sum(s == "fp" for s in c_i.values()),
               "full_tp": sum(s == "tp" for s in c_f.values()), "full_fp": sum(s == "fp" for s in c_f.values()),
               "kept_tp": sum(c_f[k] == "tp" for k in keys_i & keys_f), "kept_fp": sum(c_f[k] == "fp" for k in keys_i & keys_f),
               "added_tp": sum(c_f[k] == "tp" for k in keys_f - keys_i), "added_fp": sum(c_f[k] == "fp" for k in keys_f - keys_i),
               "removed_tp": sum(c_i[k] == "tp" for k in keys_i - keys_f), "removed_fp": sum(c_i[k] == "fp" for k in keys_i - keys_f),
               "gt_matched_ilp": len(set(p2g_i.values())), "gt_matched_full": len(set(p2g_f.values())),
               "gt_lost_in_post": len(set(p2g_i.values()) - set(p2g_f.values())),
               "gt_gained_in_post": len(set(p2g_f.values()) - set(p2g_i.values())),
               "ilp_nodes": len(ilp.node_ids), "full_nodes": len(full.node_ids)}
        row["ilp_fn"] = row["gt_edges"] - row["ilp_tp"]
        row["full_fn"] = row["gt_edges"] - row["full_tp"]
        per_volume.append(row)
        for k, v in row.items():
            if isinstance(v, int) and k not in ("gt_edges",):
                tot[k] += v
        tot["gt_edges"] += row["gt_edges"]

        # geometry of the FULL output's linker errors
        pos_f, pos_gt = um(full), um(gt)
        nn_f = nn_by_frame(full, pos_f)
        g2p_f = {g: pnode for pnode, g in p2g_f.items()}
        gt_t = gt.t_of()
        gt_parent = {int(v): int(u) for u, v in gt.edges}
        gt_has_child = {a for a, _ in gt_edges}
        gt_has_parent = {b for _, b in gt_edges}
        for (u, v) in full.edges:
            u, v = int(u), int(v)
            gu, gv = p2g_f.get(u), p2g_f.get(v)
            if gu is not None and gv is not None and (gu, gv) in gt_edges:
                continue
            src_bad = gu is not None and gu in gt_has_child
            dst_bad = gv is not None and gv in gt_has_parent
            if not (src_bad or dst_bad):
                continue
            d = pos_f[v] - pos_f[u]
            true_child = [c for (pp, c) in gt_edges if pp == gu] if gu is not None else []
            tc_det = [g2p_f[c] for c in true_child if c in g2p_f]
            d_true = min(float(np.linalg.norm(pos_f[c] - pos_f[u])) for c in tc_det) if tc_det else float("nan")
            fp_rows.append({"volume": n, "embryo": embryo_of(n), "t": int(full.t_of()[u]), "pred_u": u, "pred_v": v,
                            "gt_u": gu if gu is not None else -1, "gt_v": gv if gv is not None else -1,
                            "kind": "src_has_gt_child" if src_bad and not dst_bad else "dst_has_gt_parent" if dst_bad and not src_bad else "both",
                            "disp_um": round(float(np.linalg.norm(d)), 2), "dz_um": round(float(d[0]), 2),
                            "src_nn_um": round(nn_f.get(u, float("inf")), 2),
                            "true_child_detected": int(any(c in g2p_f for c in true_child)),
                            "dist_to_true_child_um": round(d_true, 2) if tc_det else float("nan"),
                            "chosen_is_unannotated": int(gv is None),
                            "chosen_has_other_parent": int(gv is not None and gv in gt_parent and gt_parent[gv] != gu),
                            "gt_v_true_parent_detected": int(gt_parent.get(gv) in g2p_f) if gv in gt_parent else -1,
                            "stage": "added" if ("fp", gu if src_bad else None, gv if dst_bad else None) not in c_i else "kept"})
        for (gu, gv) in gt_edges:
            if c_f.get(("tp", gu, gv)) == "tp":
                continue
            pu, pv = g2p_f.get(gu), g2p_f.get(gv)
            d = pos_gt[gv] - pos_gt[gu]
            cause = ("undetected" if pu is None or pv is None else
                     "wrong_link" if any(int(x) == pu for x, _ in full.edges) or any(int(y) == pv for _, y in full.edges) else "unlinked")
            fn_rows.append({"volume": n, "embryo": embryo_of(n), "t": gt_t[gu], "gt_u": gu, "gt_v": gv, "cause": cause,
                            "disp_um": round(float(np.linalg.norm(d)), 2), "dz_um": round(float(d[0]), 2),
                            "src_nn_um": round(nn_f.get(pu, float("inf")), 2) if pu is not None else float("nan"),
                            "was_tp_at_ilp": int(c_i.get(("tp", gu, gv)) == "tp")})

    for fname, rows in [("per_volume.csv", per_volume), ("fp_edges.csv", fp_rows), ("fn_edges.csv", fn_rows)]:
        if rows:
            with open(a.out_dir / fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    (a.out_dir / "totals.json").write_text(json.dumps(dict(tot), indent=1))

    def block(title: str, rows: list[dict]) -> None:
        c = Counter()
        for r in rows:
            for k, v in r.items():
                if isinstance(v, int):
                    c[k] += v
        print(f"\n=== {title} ({len(rows)} volumes) ===")
        print(f"  counted edges   ILP: TP={c['ilp_tp']} FP={c['ilp_fp']} FN={c['ilp_fn']}   FULL: TP={c['full_tp']} FP={c['full_fp']} FN={c['full_fn']}")
        print(f"  kept by post-processing:    TP={c['kept_tp']}  FP={c['kept_fp']}")
        print(f"  ADDED by post-processing:   TP={c['added_tp']} (fixed)   FP={c['added_fp']} (new errors)")
        print(f"  REMOVED by post-processing: TP={c['removed_tp']} (damage)  FP={c['removed_fp']} (cleaned)")
        print(f"  GT nodes matched: ILP {c['gt_matched_ilp']} -> FULL {c['gt_matched_full']}  (lost {c['gt_lost_in_post']}, gained {c['gt_gained_in_post']});"
              f" pred nodes {c['ilp_nodes']} -> {c['full_nodes']}")

    block("ALL held-out", per_volume)
    for emb in sorted({r["embryo"] for r in per_volume}):
        block(f"embryo {emb}", [r for r in per_volume if r["embryo"] == emb])

    # geometry of the full output's linker errors
    fpa = np.array([[r["disp_um"], abs(r["dz_um"]), r["src_nn_um"]] for r in fp_rows], dtype=np.float64) if fp_rows else np.zeros((0, 3))
    wl = [r for r in fn_rows if r["cause"] == "wrong_link"]
    fna = np.array([[r["disp_um"], abs(r["dz_um"]), r["src_nn_um"]] for r in wl], dtype=np.float64) if wl else np.zeros((0, 3))
    print("\n=== geometry of the FULL output's linker errors ===")
    for label, arr in [("FP edges (wrong predicted link)", fpa), ("FN edges, both detected but wrong link", fna)]:
        if len(arr):
            print(f"  {label}: n={len(arr)}  displacement median {np.median(arr[:, 0]):.1f} um (p90 {np.percentile(arr[:, 0], 90):.1f});"
                  f" |dz| median {np.median(arr[:, 1]):.1f} um, |dz|>=3.25um (2 z-planes) {100 * (arr[:, 1] >= 3.25).mean():.0f}%;"
                  f" source nn median {np.median(arr[:, 2][np.isfinite(arr[:, 2])]):.1f} um, nn<10um {100 * (arr[:, 2] < 10).mean():.0f}%")
    if fp_rows:
        kinds = Counter(r["kind"] for r in fp_rows)
        src = [r for r in fp_rows if r["gt_u"] >= 0 and r["true_child_detected"]]
        if src:
            dd = np.array([[r["disp_um"], r["dist_to_true_child_um"]] for r in src])
            print(f"  FP with source matched and true child detected: n={len(src)}; chosen child closer than true child in"
                  f" {100 * np.mean(dd[:, 0] < dd[:, 1]):.0f}% (chosen median {np.median(dd[:, 0]):.1f} um, true median {np.median(dd[:, 1]):.1f} um);"
                  f" chosen node unannotated {100 * np.mean([r['chosen_is_unannotated'] for r in src]):.0f}%")
        print(f"  FP kinds: {dict(kinds)}; true child detected (src kind): {100 * np.mean([r['true_child_detected'] for r in fp_rows if r['gt_u'] >= 0]):.0f}%;"
              f"  added by post-processing: {sum(r['stage'] == 'added' for r in fp_rows)} of {len(fp_rows)}")
    if fn_rows:
        c = Counter(r["cause"] for r in fn_rows)
        print(f"  FN by cause: {dict(c)};  FN that were TP at the ILP stage: {sum(r['was_tp_at_ilp'] for r in fn_rows)}")
    # counted-edge displacement reference: all GT edges
    print(f"\nwrote {a.out_dir}/per_volume.csv, fp_edges.csv, fn_edges.csv, totals.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

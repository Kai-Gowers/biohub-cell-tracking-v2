#!/usr/bin/env python3
"""Plain summary of the competition data: what we train and test on, how much is annotated.

    PYTHONPATH=src python scripts/dataset_summary.py --split dist/heldout_split.json --out-dir reports/figures/...

Prints markdown tables (per embryo, train vs held-out) and writes one figure: annotated cells per
frame vs. estimated cells per frame, per volume.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cell_tracking.config import SCALE  # noqa: E402
from cell_tracking.io_geff import embryo_of, list_geff_datasets, read_geff  # noqa: E402
from cell_tracking.io_zarr import read_array_meta  # noqa: E402

C = {"44b6": "#eb6834", "6bba": "#2a78d6"}
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.grid": True,
                     "grid.color": "#e6e5e0", "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight"})


def track_lengths(edges) -> list[int]:
    children = defaultdict(list)
    has_parent = set()
    for u, v in edges:
        children[int(u)].append(int(v))
        has_parent.add(int(v))
    starts = [u for u in children if u not in has_parent]
    for u, kids in children.items():
        if len(kids) >= 2:
            starts.extend(kids)
    out = []
    for s in starts:
        n, cur = 0, s
        while len(children.get(cur, [])) == 1:
            cur = children[cur][0]
            n += 1
        if n:
            out.append(n + 1)  # nodes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, default=Path("data/biohub-cell-tracking-during-development/train"))
    ap.add_argument("--test-dir", type=Path, default=Path("data/biohub-cell-tracking-during-development/test"))
    ap.add_argument("--split", type=Path, default=Path("dist/heldout_split.json"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    heldout = set(json.load(open(args.split))["datasets"])

    rows = []
    for name in list_geff_datasets(args.train_dir):
        shape, dtype = read_array_meta(args.train_dir / f"{name}.zarr")
        g = read_geff(args.train_dir / f"{name}.geff")
        T = shape[0]
        n_frames_annot = len(set(g.t.tolist()))
        deg = defaultdict(int)
        for u, _ in g.edges:
            deg[int(u)] += 1
        tl = track_lengths(g.edges)
        disp = []
        cz = {int(i): np.array([z, y, x], float) * SCALE for i, z, y, x in zip(g.node_ids, g.z, g.y, g.x)}
        for u, v in g.edges:
            disp.append(np.linalg.norm(cz[int(v)] - cz[int(u)]))
        rows.append(dict(
            name=name, embryo=embryo_of(name), heldout=name in heldout, shape=shape, dtype=str(dtype), T=T,
            n_nodes=len(g.node_ids), n_edges=len(g.edges), n_div=sum(1 for d in deg.values() if d >= 2),
            n_tracks=len(tl), track_len_median=float(np.median(tl)) if tl else 0.0,
            frames_annot=n_frames_annot, t_min=int(g.t.min()), t_max=int(g.t.max()),
            nodes_per_frame=len(g.node_ids) / n_frames_annot,
            est_nodes=g.estimated_number_of_nodes, est_per_frame=(g.estimated_number_of_nodes or 0) / T,
            disp_median=float(np.median(disp)) if disp else 0.0,
        ))
    test_names = sorted(p.stem for p in args.test_dir.glob("*.zarr"))
    test_shapes = {n: read_array_meta(args.test_dir / f"{n}.zarr")[0] for n in test_names}

    # ---- global facts
    shapes = sorted({r["shape"] for r in rows})
    print(f"train volumes: {len(rows)}   distinct shapes (T,Z,Y,X): {shapes}   dtype: {sorted({r['dtype'] for r in rows})}")
    print(f"voxel size (z,y,x) um: {SCALE.tolist()}   -> a frame is {shapes[0][1]*SCALE[0]:.0f} x {shapes[0][2]*SCALE[1]:.0f} x {shapes[0][3]*SCALE[2]:.0f} um")
    print(f"test folder: {len(test_names)} volumes {test_names}, shapes {sorted(set(test_shapes.values()))}, "
          f"all also present in train: {all((args.train_dir / f'{n}.zarr').exists() for n in test_names)}")

    def agg(rs):
        n = len(rs)
        if not n:
            return None
        return dict(
            volumes=n, frames=sum(r["T"] for r in rs), nodes=sum(r["n_nodes"] for r in rs), edges=sum(r["n_edges"] for r in rs),
            divisions=sum(r["n_div"] for r in rs), tracks=sum(r["n_tracks"] for r in rs),
            nodes_per_frame=np.mean([r["nodes_per_frame"] for r in rs]),
            nodes_per_frame_range=(min(r["nodes_per_frame"] for r in rs), max(r["nodes_per_frame"] for r in rs)),
            est_per_frame=np.mean([r["est_per_frame"] for r in rs]),
            est_per_frame_range=(min(r["est_per_frame"] for r in rs), max(r["est_per_frame"] for r in rs)),
            labelled_frac=sum(r["n_nodes"] for r in rs) / sum(r["est_nodes"] or 0 for r in rs),
            track_len_median=np.median([x for r in rs for x in [r["track_len_median"]]]),
            frames_annot=np.mean([r["frames_annot"] for r in rs]),
            disp_median=np.median([r["disp_median"] for r in rs]),
        )

    groups = {
        "all 199": rows,
        "44b6": [r for r in rows if r["embryo"] == "44b6"],
        "6bba": [r for r in rows if r["embryo"] == "6bba"],
        "train 179": [r for r in rows if not r["heldout"]],
        "held-out 20": [r for r in rows if r["heldout"]],
        "held-out 44b6": [r for r in rows if r["heldout"] and r["embryo"] == "44b6"],
        "held-out 6bba": [r for r in rows if r["heldout"] and r["embryo"] == "6bba"],
    }
    A = {k: agg(v) for k, v in groups.items()}
    cols = list(groups)
    print("\n| | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    def line(label, f):
        print(f"| {label} | " + " | ".join(f(A[c]) for c in cols) + " |")
    line("volumes", lambda a: f"{a['volumes']}")
    line("frames (100 per volume)", lambda a: f"{a['frames']}")
    line("annotated nodes", lambda a: f"{a['nodes']}")
    line("annotated edges", lambda a: f"{a['edges']}")
    line("annotated divisions", lambda a: f"{a['divisions']}")
    line("annotated tracks (chains)", lambda a: f"{a['tracks']}")
    line("median track length (nodes), per volume", lambda a: f"{a['track_len_median']:.0f}")
    line("annotated nodes per frame, mean (range over volumes)", lambda a: f"{a['nodes_per_frame']:.1f} ({a['nodes_per_frame_range'][0]:.1f}-{a['nodes_per_frame_range'][1]:.1f})")
    line("estimated true cells per frame, mean (range)", lambda a: f"{a['est_per_frame']:.0f} ({a['est_per_frame_range'][0]:.0f}-{a['est_per_frame_range'][1]:.0f})")
    line("fraction of cells annotated", lambda a: f"{100*a['labelled_frac']:.1f}%")
    line("frames with any annotation, mean of 100", lambda a: f"{a['frames_annot']:.0f}")
    line("median cell displacement per frame, um", lambda a: f"{a['disp_median']:.1f}")

    # ---- figure: annotated vs estimated cells per frame
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for e in ("44b6", "6bba"):
        rs = [r for r in rows if r["embryo"] == e]
        ax.scatter([r["est_per_frame"] for r in rs], [r["nodes_per_frame"] for r in rs], s=22, color=C[e], alpha=0.75,
                   edgecolor="none", label=f"{e} ({len(rs)} volumes)")
        ho = [r for r in rs if r["heldout"]]
        ax.scatter([r["est_per_frame"] for r in ho], [r["nodes_per_frame"] for r in ho], s=60, facecolor="none", edgecolor="#0b0b0b", lw=0.8,
                   label="held-out" if e == "44b6" else None)
    ax.set_xlabel("estimated true cells per frame (from the GEFF metadata)")
    ax.set_ylabel("annotated cells per frame")
    ax.set_title("Each volume: how many cells are there, and how many are labelled")
    ax.legend(fontsize=8)
    fig.savefig(args.out_dir / "data01_annotated_vs_estimated_cells_per_frame.png"); plt.close(fig)
    with open(args.out_dir / "dataset_rows.json", "w") as f:
        json.dump(rows, f, indent=1, default=str)
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()

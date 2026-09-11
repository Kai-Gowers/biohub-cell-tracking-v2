#!/usr/bin/env python
"""Compare the ported pipeline against the reference-pack oracle.

Mode ``graph`` (default): two ``.geff`` directories (ours vs oracle) -- for
every stem present in both, compare the node set (exact integer coordinates
per frame) and the edge set (as coordinate pairs, so node ids need not
match), and the edge_prob values on shared edges.

    python scripts/parity/parity_check.py --ours dist/parity/ilp --oracle dist/ref_pack/patched

Mode ``postprocess``: run our ``filter_output_graph`` on the oracle ``.geff``
graphs and compare against the notebook's own cell-18 functions executed from
a scratch extraction (``--notebook-module``), row by row.

    python scripts/parity/parity_check.py --mode postprocess --oracle dist/ref_pack/patched \
        --deepcenter context/pack_deepcenter/weights/full_frame_center/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cell_tracking.io_geff import read_geff  # noqa: E402


def _node_keys(g) -> dict[tuple, int]:
    """(t, z, y, x) rounded to 3 decimals -> node id."""
    keys = {}
    for i in range(len(g.node_ids)):
        keys[(int(g.t[i]), round(float(g.z[i]), 3), round(float(g.y[i]), 3), round(float(g.x[i]), 3))] = int(g.node_ids[i])
    return keys


def _edge_keys(g) -> dict[tuple, float]:
    by_id = {int(g.node_ids[i]): (int(g.t[i]), round(float(g.z[i]), 3), round(float(g.y[i]), 3), round(float(g.x[i]), 3)) for i in range(len(g.node_ids))}
    out = {}
    for k, (u, v) in enumerate(g.edges.tolist()):
        p = float(g.edge_prob[k]) if g.edge_prob is not None else float("nan")
        out[(by_id[int(u)], by_id[int(v)])] = p
    return out


def compare_graphs(ours, oracle, label: str) -> dict:
    n_ours, n_ref = _node_keys(ours), _node_keys(oracle)
    common_nodes = set(n_ours) & set(n_ref)
    e_ours, e_ref = _edge_keys(ours), _edge_keys(oracle)
    common_edges = set(e_ours) & set(e_ref)
    union_edges = set(e_ours) | set(e_ref)
    prob_diff = [abs(e_ours[k] - e_ref[k]) for k in common_edges if np.isfinite(e_ours[k]) and np.isfinite(e_ref[k])]
    rep = {
        "label": label,
        "nodes_ours": len(n_ours), "nodes_oracle": len(n_ref), "nodes_common": len(common_nodes),
        "nodes_only_ours": len(set(n_ours) - set(n_ref)), "nodes_only_oracle": len(set(n_ref) - set(n_ours)),
        "edges_ours": len(e_ours), "edges_oracle": len(e_ref), "edges_common": len(common_edges),
        "edge_jaccard": len(common_edges) / max(len(union_edges), 1),
        "edge_prob_max_abs_diff": max(prob_diff) if prob_diff else None,
        "edge_prob_mean_abs_diff": float(np.mean(prob_diff)) if prob_diff else None,
    }
    rep["node_sets_identical"] = rep["nodes_only_ours"] == 0 and rep["nodes_only_oracle"] == 0
    return rep


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["graph", "postprocess"], default="graph")
    p.add_argument("--ours", type=Path, help="our .geff dir (graph mode)")
    p.add_argument("--oracle", type=Path, required=True, help="oracle .geff dir")
    p.add_argument("--notebook-module", type=Path, default=None,
                   help="postprocess mode: scratch module exposing the notebook's filter_output_graph")
    p.add_argument("--deepcenter", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/biohub-cell-tracking-during-development/train")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    reports = []
    if args.mode == "graph":
        stems = sorted({g.stem for g in args.ours.glob("*.geff")} & {g.stem for g in args.oracle.glob("*.geff")})
        if not stems:
            raise SystemExit("no common stems")
        for stem in stems:
            rep = compare_graphs(read_geff(args.ours / f"{stem}.geff"), read_geff(args.oracle / f"{stem}.geff"), stem)
            reports.append(rep)
            print(json.dumps(rep, indent=1))
    else:
        from cell_tracking.models.deepcenter import load_deepcenter
        from cell_tracking.pack_predict import PackVolume
        from cell_tracking.pipeline import geff_to_nodes_edges, nodes_edges_to_graph
        from cell_tracking.postprocess import PostprocessConfig, filter_output_graph

        import importlib.util
        spec = importlib.util.spec_from_file_location("nb_post", args.notebook_module)
        nb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nb)

        dc = load_deepcenter(args.deepcenter) if args.deepcenter else None
        for gpath in sorted(args.oracle.glob("*.geff")):
            stem = gpath.stem
            g = read_geff(gpath)
            vol = PackVolume(args.data_dir / f"{stem}.zarr")
            nodes, edges = geff_to_nodes_edges(g)
            ours_nodes, ours_edges, _ = filter_output_graph(
                nodes, edges, cfg=PostprocessConfig(), frame_source=vol.raw_frame, deepcenter=dc, name=stem,
            )
            ours_graph, ours_prob = nodes_edges_to_graph(ours_nodes, ours_edges)
            nb_nodes, nb_edges = nb.run_notebook_postprocess(gpath, stem, args.data_dir, args.deepcenter)
            ref_graph, ref_prob = nodes_edges_to_graph(nb_nodes, nb_edges)
            from cell_tracking.io_geff import GeffGraph
            def to_geff(graph, prob):
                a = graph.to_arrays()
                return GeffGraph(a["node_ids"], a["t"], a["z"], a["y"], a["x"], a["edges"], None, prob)
            rep = compare_graphs(to_geff(ours_graph, ours_prob), to_geff(ref_graph, ref_prob), stem)
            reports.append(rep)
            print(json.dumps(rep, indent=1))

    ok = all(r["node_sets_identical"] and r["edge_jaccard"] >= 0.99 for r in reports)
    print("PARITY", "OK" if ok else "MISMATCH")
    if args.json_out:
        args.json_out.write_text(json.dumps(reports, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Global graph selection with the tracksdata ILP solver (ported).

Mirrors ``build_graph`` and the ``--use-ilp`` branch of the pilkwang pack's
``scripts/predict_unet_transformer.py`` (retrieved 2026-09-11). The solver
(``tracksdata.solvers.ILPSolver``, SCIP through ``ilpy``) has one binary
variable per node / appearance / disappearance / division and per edge, with
flow conservation ``appear + sum(in) = node`` and
``disappear + sum(out) = node + division``, ``node >= division``, and objective
``sum(appearance_w * appear + disappearance_w * disappear + division_w * division)
+ sum(edge_w * edge_prob * edge)``. The 0.942 notebook runs it with
``edge_w = -1.0, appearance = 0.0, disappearance = 2.0, division = 1.2``.

The ILP's *edges* are later thrown away by the post-processing motion relink;
what survives is the selected node set and the ``edge_prob`` lookup table.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ILPConfig:
    edge_weight: float = -1.0
    appearance_weight: float = 0.0
    disappearance_weight: float = 2.0
    division_weight: float = 1.2


@contextlib.contextmanager
def _suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _import_tracksdata():
    try:
        import polars as pl
        import tracksdata as td
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tracksdata/polars are required for the ILP stage: "
            "pip install --no-index --find-links context/pack_primary/wheels '.[ilp]'"
        ) from exc
    return td, pl


def build_graph(coords: np.ndarray, edges: list[tuple[int, int, float, float]]):
    """tracksdata InMemoryGraph from (N,4) [t,z,y,x] coords and (src,tgt,prob,dist) edges.

    Node ids are tracksdata's sequential ids in ``coords`` order.
    """
    td, pl = _import_tracksdata()
    graph = td.graph.InMemoryGraph()
    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in coords
    ])
    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {
                "source_id": node_ids[src],
                "target_id": node_ids[tgt],
                "edge_prob": float(prob),
                "edge_dist": float(dist),
            }
            for src, tgt, prob, dist in edges
        ])
    return graph, node_ids


def graph_to_arrays(graph) -> tuple[dict[int, dict], list[dict]]:
    """``(nodes_by_id, edges)`` in the post-processing's in-memory format."""
    nodes_by_id: dict[int, dict] = {}
    for row in graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"]).iter_rows(named=True):
        nid = int(row["node_id"])
        nodes_by_id[nid] = {
            "node_id": nid,
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
    edges: list[dict] = []
    if graph.num_edges() > 0:
        keys = ["source_id", "target_id"]
        if "edge_prob" in graph.edge_attr_keys():
            keys.append("edge_prob")
        for row in graph.edge_attrs(attr_keys=keys).iter_rows(named=True):
            prob = row.get("edge_prob")
            edges.append({
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if prob is None else float(prob),
            })
    return nodes_by_id, edges


def solve_ilp(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
    cfg: ILPConfig = ILPConfig(),
    *,
    use_ilp: bool = True,
    verbose: bool = False,
) -> tuple[dict[int, dict], list[dict], dict]:
    """Build the candidate graph and (optionally) solve the ILP.

    Returns ``(nodes_by_id, edges, stats)``. With ``use_ilp=False`` (or no
    candidate edges, as the pack does) the raw candidate graph is returned.
    """
    td, _ = _import_tracksdata()
    graph, _ = build_graph(coords, edges)
    stats = {"ilp_input_nodes": graph.num_nodes(), "ilp_input_edges": graph.num_edges(), "ilp_solved": 0}
    if use_ilp and graph.num_edges() > 0:
        solver = td.solvers.ILPSolver(
            edge_weight=cfg.edge_weight * td.EdgeAttr("edge_prob"),
            appearance_weight=cfg.appearance_weight,
            disappearance_weight=cfg.disappearance_weight,
            division_weight=cfg.division_weight,
        )
        if verbose:
            graph = solver.solve(graph)
        else:
            with _suppress_output():
                graph = solver.solve(graph)
        stats["ilp_solved"] = 1
    nodes_by_id, out_edges = graph_to_arrays(graph)
    stats["ilp_output_nodes"] = len(nodes_by_id)
    stats["ilp_output_edges"] = len(out_edges)
    return nodes_by_id, out_edges, stats

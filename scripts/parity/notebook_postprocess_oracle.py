"""Execute the 0.942 notebook's own post-processing (cells 8, 12, 18) as an oracle.

Loaded by ``parity_check.py --mode postprocess`` via ``--notebook-module``.
Cell 8's environment variables are applied, cell 12's constant block is
executed (with the DeepCenter checkpoint pointed at ``context/pack_deepcenter``),
and cell 18's function definitions are executed verbatim up to the point
where the notebook starts writing ``submission.csv``. ``TEST_DIR`` is rebound
to the data directory the way the notebook's validator does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "context" / "biohub-0-942-lb-proxy-score-0-9417.ipynb"

_NS: dict | None = None


def _cell(index: int) -> str:
    nb = json.loads(NOTEBOOK.read_text())
    return "".join(nb["cells"][index]["source"])


def _namespace(data_dir: Path, deepcenter_path: Path | None) -> dict:
    global _NS
    if _NS is not None:
        return _NS
    cell8 = _cell(8)
    # cell 8 = env-var assignments + prints; run it (it has no side effects beyond os.environ)
    ns: dict = {"__name__": "__nb__"}
    exec(compile(cell8, "<cell8>", "exec"), ns)
    if deepcenter_path is not None:
        os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(deepcenter_path)
        os.environ["BIOHUB_DEEPCENTER_CHECKPOINT_DEFAULT"] = str(deepcenter_path)
        os.environ["BIOHUB_DEEPCENTER_MANIFEST_DEFAULT"] = str(REPO_ROOT / "context" / "pack_deepcenter" / "ARTIFACT_MANIFEST.json")
    else:
        os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "0"
        os.environ["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = "0"
    cell12 = _cell(12)
    cell12 = cell12.replace("from IPython.display import display", "display = print")
    exec(compile(cell12, "<cell12>", "exec"), ns)
    ns["TEST_DIR"] = Path(data_dir)
    cell18 = _cell(18)
    cut = cell18.index("DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()")
    exec(compile(cell18[:cut], "<cell18-functions>", "exec"), ns)
    ns["DEEPCENTER_VETO_DETECTOR"] = ns["load_deepcenter_veto_detector"]()
    _NS = ns
    return ns


def run_notebook_postprocess(geff_path: Path, stem: str, data_dir: Path, deepcenter_path: Path | None):
    """The notebook's per-dataset loop body: load .geff via tracksdata, run filter_output_graph."""
    ns = _namespace(Path(data_dir), Path(deepcenter_path) if deepcenter_path else None)
    graph = ns["graph_from_geff"](Path(geff_path))
    nodes_by_id: dict[int, dict] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes_by_id[node_id] = {"node_id": node_id, "t": int(row["t"]), "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])}
    raw_edges: list[dict] = []
    for row in graph.edge_attrs().iter_rows(named=True):
        edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
        raw_edges.append({"source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                          "edge_prob": None if edge_prob is None else float(edge_prob)})
    nodes_by_id, edges, stats = ns["filter_output_graph"](nodes_by_id, raw_edges, dataset=stem, deepcenter_bundle=ns["DEEPCENTER_VETO_DETECTOR"])
    return nodes_by_id, edges

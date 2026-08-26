"""Convert predicted `.geff` graphs into `submission.csv`.

Rows stream straight to disk rather than accumulating in a DataFrame, and the
writer refuses to finish if any expected dataset is missing -- a submission
silently short one video scores zero on that video's share rather than
erroring, so it is worth catching here.
"""

from __future__ import annotations

import csv
from pathlib import Path

from cell_tracking.io_geff import read_geff

COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]
FILL = -1


def write_submission(
    geff_dir: Path | str,
    out_path: Path | str,
    *,
    expected: list[str] | None = None,
) -> dict:
    """Stream every `<geff_dir>/*.geff` into one submission CSV."""
    geff_dir, out_path = Path(geff_dir), Path(out_path)
    names = sorted(p.stem for p in geff_dir.glob("*.geff"))
    if expected is not None:
        missing = sorted(set(expected) - set(names))
        if missing:
            raise ValueError(
                f"No predicted graph for {missing}. Every hidden-test dataset must appear "
                f"in submission.csv; found {names}."
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_id = 0
    per_dataset: dict[str, dict[str, int]] = {}

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for name in names:
            graph = read_geff(geff_dir / f"{name}.geff")
            n_nodes = n_edges = 0
            for i in range(len(graph.node_ids)):
                writer.writerow(
                    [
                        row_id,
                        name,
                        "node",
                        int(graph.node_ids[i]),
                        int(graph.t[i]),
                        int(round(float(graph.z[i]))),
                        int(round(float(graph.y[i]))),
                        int(round(float(graph.x[i]))),
                        FILL,
                        FILL,
                    ]
                )
                row_id += 1
                n_nodes += 1
            for u, v in graph.edges:
                writer.writerow(
                    [row_id, name, "edge", FILL, FILL, FILL, FILL, FILL, int(u), int(v)]
                )
                row_id += 1
                n_edges += 1
            per_dataset[name] = {"nodes": n_nodes, "edges": n_edges}

    return {
        "path": str(out_path),
        "rows": row_id,
        "datasets": len(names),
        "per_dataset": per_dataset,
    }

"""Mutable tracking graph shared by the ILP / post-processing stages, the scorer and the writers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cell_tracking.config import SCALE


@dataclass
class Node:
    node_id: int
    t: int
    zyx: np.ndarray  # raw-volume voxel coordinates, float
    score: float = 1.0

    @property
    def um(self) -> np.ndarray:
        return self.zyx * SCALE


@dataclass
class TrackGraph:
    """Directed graph of detections; edges always point forward in time."""

    nodes: dict[int, Node] = field(default_factory=dict)
    edges: set[tuple[int, int]] = field(default_factory=set)
    _next_id: int = 1

    # --- construction -----------------------------------------------------
    def add_node(self, t: int, zyx: np.ndarray, score: float = 1.0) -> int:
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Node(nid, int(t), np.asarray(zyx, dtype=np.float64), float(score))
        return nid

    def add_edge(self, u: int, v: int) -> None:
        if u not in self.nodes or v not in self.nodes:
            raise KeyError(f"edge ({u}, {v}) references a missing node")
        if self.nodes[v].t <= self.nodes[u].t:
            raise ValueError(
                f"edge ({u}, {v}) is not forward in time: "
                f"t={self.nodes[u].t} -> t={self.nodes[v].t}"
            )
        self.edges.add((u, v))

    @classmethod
    def from_arrays(
        cls,
        node_ids,
        t,
        zyx,
        edges,
        scores=None,
    ) -> "TrackGraph":
        """Build a graph that keeps the caller's node ids.

        Post-processing assigns synthetic node ids (``max + 1``) that must
        survive into the ``.geff`` / CSV rows, so ``add_node``'s auto-ids
        cannot be used there.
        """
        node_ids = np.asarray(node_ids, dtype=np.int64)
        t = np.asarray(t, dtype=np.int64)
        zyx = np.asarray(zyx, dtype=np.float64).reshape(-1, 3)
        if len(node_ids) != len(set(node_ids.tolist())):
            raise ValueError("node_ids must be unique")
        g = cls()
        for i, nid in enumerate(node_ids.tolist()):
            score = 1.0 if scores is None else float(scores[i])
            g.nodes[nid] = Node(nid, int(t[i]), zyx[i].copy(), score)
        g._next_id = int(node_ids.max()) + 1 if len(node_ids) else 1
        for u, v in np.asarray(edges, dtype=np.int64).reshape(-1, 2).tolist():
            g.add_edge(int(u), int(v))
        return g

    # --- queries ----------------------------------------------------------
    def out_degree(self) -> dict[int, int]:
        deg: dict[int, int] = {}
        for u, _ in self.edges:
            deg[u] = deg.get(u, 0) + 1
        return deg

    def in_degree(self) -> dict[int, int]:
        deg: dict[int, int] = {}
        for _, v in self.edges:
            deg[v] = deg.get(v, 0) + 1
        return deg

    def track_starts(self) -> list[int]:
        """Nodes with no predecessor."""
        has_pred = {v for _, v in self.edges}
        return sorted(n for n in self.nodes if n not in has_pred)

    def n_divisions(self) -> int:
        return sum(1 for d in self.out_degree().values() if d >= 2)

    def validate(self) -> None:
        """Assert the invariants the competition format requires.

        Every submitted edge must be dt=1: all ground-truth edges are, so a
        longer edge is guaranteed not to score. Linking only ever connects
        consecutive frames, so this is a sanity check rather than something
        that can fail.
        """
        for u, v in self.edges:
            if u not in self.nodes or v not in self.nodes:
                raise ValueError(f"edge ({u}, {v}) references a missing node")
            dt = self.nodes[v].t - self.nodes[u].t
            if dt != 1:
                raise ValueError(f"edge ({u}, {v}) spans dt={dt}, expected 1")
        for v, d in self.in_degree().items():
            if d > 1:
                raise ValueError(f"node {v} has in-degree {d}, expected <= 1")

    def to_arrays(self) -> dict[str, np.ndarray]:
        ids = sorted(self.nodes)
        return {
            "node_ids": np.array(ids, dtype=np.int64),
            "t": np.array([self.nodes[i].t for i in ids], dtype=np.int64),
            "z": np.array([self.nodes[i].zyx[0] for i in ids], dtype=np.float64),
            "y": np.array([self.nodes[i].zyx[1] for i in ids], dtype=np.float64),
            "x": np.array([self.nodes[i].zyx[2] for i in ids], dtype=np.float64),
            "edges": (
                np.array(sorted(self.edges), dtype=np.int64)
                if self.edges
                else np.zeros((0, 2), dtype=np.int64)
            ),
        }

    def summary(self) -> str:
        return (
            f"{len(self.nodes)} nodes, {len(self.edges)} edges, "
            f"{self.n_divisions()} divisions, {len(self.track_starts())} track starts"
        )

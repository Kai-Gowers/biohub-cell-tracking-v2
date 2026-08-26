"""Read sparse GEFF tracking graphs from competition train layout."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GeffGraph:
    """Sparse tracking graph for one sample."""

    node_ids: np.ndarray  # (N,)
    t: np.ndarray  # (N,) int
    z: np.ndarray  # (N,)
    y: np.ndarray  # (N,)
    x: np.ndarray  # (N,)
    edges: np.ndarray  # (E, 2) source -> target node ids
    estimated_number_of_nodes: int | None = None

    def centroids_by_t(self) -> dict[int, list[tuple[int, int, int]]]:
        out: dict[int, list[tuple[int, int, int]]] = {}
        for i in range(len(self.node_ids)):
            ti = int(self.t[i])
            out.setdefault(ti, []).append(
                (int(self.z[i]), int(self.y[i]), int(self.x[i]))
            )
        return out

    def nodes_by_t(self) -> dict[int, list[tuple[int, tuple[int, int, int]]]]:
        """Map t -> list of (node_id, (z,y,x))."""
        out: dict[int, list[tuple[int, tuple[int, int, int]]]] = {}
        for i in range(len(self.node_ids)):
            ti = int(self.t[i])
            nid = int(self.node_ids[i])
            out.setdefault(ti, []).append(
                (nid, (int(self.z[i]), int(self.y[i]), int(self.x[i])))
            )
        return out

    def t_of(self) -> dict[int, int]:
        return {int(n): int(tt) for n, tt in zip(self.node_ids, self.t)}

    def coords_of(self) -> dict[int, tuple[float, float, float]]:
        return {
            int(self.node_ids[i]): (float(self.z[i]), float(self.y[i]), float(self.x[i]))
            for i in range(len(self.node_ids))
        }

    def out_degree(self) -> dict[int, int]:
        deg: dict[int, int] = {}
        for u, _v in self.edges:
            deg[int(u)] = deg.get(int(u), 0) + 1
        return deg

    def divisions(self) -> set[int]:
        """Node ids with >= 2 outgoing edges (the metric's definition of a fork)."""
        return {n for n, d in self.out_degree().items() if d >= 2}

    def edges_by_frame_pair(self) -> dict[tuple[int, int], list[tuple[int, int]]]:
        """Map (t_source, t_target) -> list of (source_id, target_id)."""
        tmap = self.t_of()
        out: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for u, v in self.edges:
            u, v = int(u), int(v)
            key = (tmap[u], tmap[v])
            out.setdefault(key, []).append((u, v))
        return out


def list_geff_datasets(train_dir: Path | str) -> list[str]:
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    return sorted(
        d.replace(".geff", "")
        for d in os.listdir(train_dir)
        if d.endswith(".geff")
    )


def split_dataset_names(
    names: list[str], *, val_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Deterministically split volume names into (train, val) by whole volume.

    Splitting by volume (not by individual crop/frame-pair) avoids leaking
    near-duplicate adjacent timepoints from the same volume across the
    train/val boundary, and keeps the split reproducible across runs/epochs
    (same seed -> same split) independent of the global `random` module
    state used elsewhere for time/patch sampling.
    """
    ordered = sorted(names)
    shuffled = ordered[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac)) if shuffled else 0
    val_names = sorted(shuffled[:n_val])
    train_names = sorted(shuffled[n_val:])
    return train_names, val_names


def _prop_values(prop: dict | np.ndarray) -> np.ndarray:
    if isinstance(prop, dict):
        return np.asarray(prop["values"])
    return np.asarray(prop)


def _estimated_nodes(geff_path: Path, metadata: object | None = None) -> int | None:
    if metadata is not None:
        extra = getattr(metadata, "extra", None) or {}
        if isinstance(extra, dict) and "estimated_number_of_nodes" in extra:
            return int(extra["estimated_number_of_nodes"])
    for meta_name in ("zarr.json", ".zattrs"):
        meta_path = geff_path / meta_name
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        estimated = meta.get("estimated_number_of_nodes")
        if estimated is None and isinstance(meta.get("attributes"), dict):
            estimated = meta["attributes"].get("estimated_number_of_nodes")
        if estimated is not None:
            return int(estimated)
    return None


def _load_array_zarr(path: Path) -> np.ndarray:
    """Load a zarr array path (directory) with compression handled by zarr."""
    import zarr

    arr = zarr.open(str(path), mode="r")
    return np.asarray(arr[:])


def _decode_chunk_bytes(raw: bytes, meta: dict) -> bytes:
    """Best-effort decompress a single zarr v3 chunk using declared codecs."""
    codecs = meta.get("codecs") or meta.get("compressor") or []
    # zarr v3 often lists codecs as a list of dicts; apply in reverse for decode.
    codec_list = codecs if isinstance(codecs, list) else [codecs]
    data = raw
    for codec in reversed(codec_list):
        if not isinstance(codec, dict):
            continue
        name = str(codec.get("name", codec.get("id", ""))).lower()
        if name in {"bytes", "endian", "crc32c", "transpose"}:
            continue
        if "blosc" in name:
            import blosc2

            data = blosc2.decompress(data)
        elif "zstd" in name:
            import zstandard as zstd

            data = zstd.ZstdDecompressor().decompress(data)
        elif name in {"gzip", "zlib"}:
            import gzip

            data = gzip.decompress(data)
    # If nothing matched, try blosc2 then zstd.
    if data is raw:
        try:
            import blosc2

            data = blosc2.decompress(raw)
        except Exception:
            try:
                import zstandard as zstd

                data = zstd.ZstdDecompressor().decompress(raw)
            except Exception:
                data = raw
    return data


def _load_array_manual(path: Path) -> np.ndarray:
    """Fallback reader when geff/zarr APIs are unavailable."""
    if path.is_file():
        if path.suffix == ".npy":
            return np.load(path)
        if path.suffix == ".json":
            with open(path) as f:
                return np.asarray(json.load(f))

    values = path / "values"
    if values.exists() and path.name != "values":
        return _load_array_manual(values)

    meta_path = path / "zarr.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not load GEFF array at {path}")

    with open(meta_path) as f:
        meta = json.load(f)
    shape = tuple(meta.get("shape", []))
    dtype = np.dtype(meta.get("data_type", meta.get("dtype", "float64")))
    expected = int(np.prod(shape)) if shape else None

    candidates: list[Path] = []
    flat = path / "0"
    if flat.is_file():
        candidates.append(flat)
    chunk0 = path / "c" / "0"
    if chunk0.is_file():
        candidates.append(chunk0)
    for leaf in sorted(path.rglob("*")):
        if leaf.is_file() and leaf.name not in {"zarr.json", ".zarray", ".zattrs", ".zgroup"}:
            candidates.append(leaf)

    last_err: Exception | None = None
    for leaf in candidates:
        raw = leaf.read_bytes()
        # Uncompressed raw dump
        try:
            arr = np.frombuffer(raw, dtype=dtype)
            if expected is None or arr.size == expected:
                return arr.reshape(shape) if shape else arr
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        # Compressed chunk
        try:
            decoded = _decode_chunk_bytes(raw, meta)
            arr = np.frombuffer(decoded, dtype=dtype)
            if expected is None or arr.size == expected:
                return arr.reshape(shape) if shape else arr
        except Exception as exc:  # noqa: BLE001
            last_err = exc

    raise ValueError(f"Failed to decode GEFF array at {path}: {last_err}")


def _load_array(path: Path) -> np.ndarray:
    """Load an array from a GEFF/zarr path (compressed chunks supported)."""
    path = Path(path)
    try:
        return _load_array_zarr(path if path.is_dir() else path.parent)
    except Exception:
        pass
    return _load_array_manual(path)


def _read_prop(geff_path: Path, name: str) -> np.ndarray:
    candidates = [
        geff_path / "nodes" / "props" / name / "values",
        geff_path / "nodes" / "props" / name,
        geff_path / "nodes" / name,
    ]
    for cand in candidates:
        if cand.exists():
            return _load_array(cand)
    raise FileNotFoundError(f"Missing GEFF node prop {name!r} under {geff_path}")


def read_geff(geff_path: Path | str) -> GeffGraph:
    """Load a competition `.geff` graph."""
    geff_path = Path(geff_path)
    if not geff_path.exists():
        raise FileNotFoundError(geff_path)

    # Preferred: official geff reader (handles zstd/blosc zarr codecs).
    try:
        from geff.core_io import read_to_memory

        mem = read_to_memory(
            str(geff_path),
            structure_validation=False,
            node_props=["t", "z", "y", "x"],
            edge_props=[],
        )
        node_ids = np.asarray(mem["node_ids"])
        props = mem["node_props"]
        t = _prop_values(props["t"]).astype(np.int64)
        z = _prop_values(props["z"])
        y = _prop_values(props["y"])
        x = _prop_values(props["x"])
        edges = np.asarray(mem["edge_ids"])
        if edges.ndim == 1:
            edges = edges.reshape(-1, 2)
        edges = edges.astype(np.int64)
        return GeffGraph(
            node_ids=node_ids.astype(np.int64),
            t=t,
            z=np.asarray(z),
            y=np.asarray(y),
            x=np.asarray(x),
            edges=edges,
            estimated_number_of_nodes=_estimated_nodes(geff_path, mem.get("metadata")),
        )
    except ImportError:
        pass
    except Exception:
        # Fall through to manual/zarr path loaders.
        pass

    ids_path = geff_path / "nodes" / "ids"
    node_ids = _load_array(ids_path) if ids_path.exists() else None
    t = _read_prop(geff_path, "t").astype(np.int64)
    z = _read_prop(geff_path, "z")
    y = _read_prop(geff_path, "y")
    x = _read_prop(geff_path, "x")
    if node_ids is None:
        node_ids = np.arange(len(t), dtype=np.int64)
    else:
        node_ids = node_ids.astype(np.int64)

    edges_path = geff_path / "edges" / "ids"
    if edges_path.exists():
        edges = _load_array(edges_path).astype(np.int64)
        if edges.ndim == 1:
            edges = edges.reshape(-1, 2)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)

    return GeffGraph(
        node_ids=node_ids,
        t=t,
        z=np.asarray(z),
        y=np.asarray(y),
        x=np.asarray(x),
        edges=edges,
        estimated_number_of_nodes=_estimated_nodes(geff_path),
    )


# --- Writing --------------------------------------------------------------
#
# The notes keep prediction (`.geff` per video) separate from CSV conversion,
# so the graph repair and diagnostics stay inspectable. These graphs are an
# intermediate artifact; they only need to round-trip through `read_geff`.


def write_geff(
    geff_path: Path | str,
    *,
    node_ids: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    edges: np.ndarray,
    estimated_number_of_nodes: int | None = None,
) -> Path:
    """Write a GEFF v1.1 graph (zarr v3 group) mirroring the competition layout."""
    import zarr

    geff_path = Path(geff_path)
    if geff_path.exists():
        import shutil

        shutil.rmtree(geff_path)
    geff_path.mkdir(parents=True)

    node_ids = np.asarray(node_ids, dtype=np.int64)
    t = np.asarray(t, dtype=np.int64)
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)

    root = zarr.open_group(str(geff_path), mode="w", zarr_format=3)
    nodes = root.create_group("nodes")
    nodes.create_array("ids", shape=node_ids.shape, dtype="int64")[:] = node_ids
    props = nodes.create_group("props")
    for name, values in (("t", t), ("z", z), ("y", y), ("x", x)):
        grp = props.create_group(name)
        grp.create_array("values", shape=values.shape, dtype=str(values.dtype))[:] = values

    edge_grp = root.create_group("edges")
    edge_grp.create_array("ids", shape=edges.shape, dtype="int64")[:] = edges
    edge_grp.create_group("props")

    def _axis(name: str, values: np.ndarray, scale: float) -> dict:
        return {
            "name": name,
            "type": "time" if name == "t" else "space",
            "unit": None,
            "min": float(values.min()) if values.size else 0.0,
            "max": float(values.max()) if values.size else 0.0,
            "scale": scale,
            "scaled_unit": None,
            "offset": None,
        }

    extra: dict[str, object] = {}
    if estimated_number_of_nodes is not None:
        extra["estimated_number_of_nodes"] = int(estimated_number_of_nodes)

    root.attrs["geff"] = {
        "geff_version": "1.1",
        "directed": True,
        "axes": [
            _axis("t", t, 1.0),
            _axis("z", z, 1.625),
            _axis("y", y, 0.40625),
            _axis("x", x, 0.40625),
        ],
        "node_props_metadata": {
            n: {"identifier": n, "dtype": d, "varlength": False}
            for n, d in (("t", "int64"), ("z", "float64"), ("y", "float64"), ("x", "float64"))
        },
        "edge_props_metadata": {},
        "extra": extra,
    }
    return geff_path

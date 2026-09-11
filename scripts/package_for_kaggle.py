#!/usr/bin/env python3
"""Zip `src/` + `scripts/` + the model files for upload as a Kaggle Dataset.

    python scripts/package_for_kaggle.py \
        --primary dist/models/pack_s0_best.pt --secondary dist/models/pack_s314159_best.pt \
        --deepcenter dist/models/deepcenter/best.pt

Without arguments it falls back to the public pack weights under context/pack_*
(so a submission can be built before our own training finishes). Writes an
`ARTIFACT_MANIFEST.json` with the sha256 of every model file so a run can
assert which weights it loaded -- the sibling repo had five submissions
interpreted against the wrong assumption about which weights were in play.

The ILP stack (tracksdata, ilpy, pyscipopt, ...) is NOT in this zip: attach
`context/pack_primary/wheels/` as a second Kaggle Dataset (`biohub-tracking-wheels`);
`notebooks/kaggle_run.ipynb` installs from it offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_OUT = REPO_ROOT / "dist" / "cell_tracking_src.zip"
PACK_PRIMARY = REPO_ROOT / "context/pack_primary/weights/unet_transformer/split_0/edge_predictor_best.pth"
PACK_SECONDARY = REPO_ROOT / "context/pack_secondary/weights/unet_transformer/split_0/edge_predictor_best.pth"
PACK_DEEPCENTER = REPO_ROOT / "context/pack_deepcenter/weights/full_frame_center/best.pt"
ILP_PACKAGES = ["tracksdata", "ilpy", "pyscipopt", "polars", "polars_runtime_32", "rustworkx", "sqlalchemy",
                "dask", "imagecodecs", "pyarrow"]

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "parity"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_tree(zf: zipfile.ZipFile, root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.suffix in {".pyc", ".pyo"}:
            continue
        if SKIP_DIRS & set(path.parts) or any(p.endswith(".egg-info") for p in path.parts):
            continue
        zf.write(path, path.relative_to(REPO_ROOT))


def _model_entry(path: Path, arcname: str) -> dict:
    entry: dict[str, object] = {"path": arcname, "source": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
    try:
        import torch

        meta = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(meta, dict) and "state_dict" in meta:
            entry["epoch"] = meta.get("epoch")
            entry["config"] = meta.get("config")
            entry["val_names"] = meta.get("val_names")
            entry["selected_by"] = meta.get("selected_by")
        elif isinstance(meta, dict) and "model_state" in meta:
            entry["epoch"] = meta.get("epoch")
            entry["config"] = meta.get("config")
    except Exception as exc:  # noqa: BLE001
        entry["metadata_error"] = str(exc)
    return entry


def package(out_path: Path, *, primary: Path, secondary: Path | None, deepcenter: Path | None) -> Path:
    if not SRC_DIR.is_dir():
        raise FileNotFoundError(f"Missing source tree: {SRC_DIR}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    manifest: dict[str, object] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": (
            "TemporalUNet3D + node transformer (pack port) -> 8-view TTA, dual-seed blend, "
            "bidirectional harmonic fusion -> tracksdata ILP -> notebook post-processing"
        ),
        "models": {},
        "ilp_packages": ILP_PACKAGES,
    }
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _add_tree(zf, SRC_DIR)
        _add_tree(zf, SCRIPTS_DIR)
        for key, path in (("primary", primary), ("secondary", secondary), ("deepcenter", deepcenter)):
            if path is None:
                continue
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"{key} model not found: {path}")
            arc = f"models/{key}{path.suffix}"
            manifest["models"][key] = _model_entry(path, arc)
            zf.write(path, arc)
            cfg = path.parent / "config.json"
            if cfg.exists():
                zf.write(cfg, f"models/{key}_config.json")
                manifest["models"][key]["config_json"] = f"models/{key}_config.json"
            print(f"Included {key}: {path}  sha256={manifest['models'][key]['sha256'][:16]}...")
        zf.writestr("ARTIFACT_MANIFEST.json", json.dumps(manifest, indent=1, default=str))
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("Upload as Kaggle Dataset `cell-tracking-src`; attach it, `biohub-tracking-wheels`")
    print("(= context/pack_primary/wheels) and the competition data; run notebooks/kaggle_run.ipynb.")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--primary", type=Path, default=PACK_PRIMARY)
    parser.add_argument("--secondary", type=Path, default=PACK_SECONDARY)
    parser.add_argument("--deepcenter", type=Path, default=PACK_DEEPCENTER)
    parser.add_argument("--no-secondary", action="store_true")
    parser.add_argument("--no-deepcenter", action="store_true")
    args = parser.parse_args()
    package(args.out, primary=args.primary, secondary=None if args.no_secondary else args.secondary,
            deepcenter=None if args.no_deepcenter else args.deepcenter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Zip `src/` + `scripts/` (+ the trained checkpoint) for upload as a Kaggle Dataset.

Writes an `ARTIFACT_MANIFEST.json` alongside the code so a run can assert
which weights it loaded.
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
DEFAULT_MODELS = REPO_ROOT / "dist" / "models"
# Duplicated rather than imported from config: this script must run from a bare
# checkout with nothing installed, which is how it is used before packaging.
CHECKPOINT_NAME = "detector.pt"
BEST_CHECKPOINT_NAME = "detector_best.pt"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}


def _add_tree(zf: zipfile.ZipFile, root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.suffix in {".pyc", ".pyo"}:
            continue
        if SKIP_DIRS & set(path.parts) or any(p.endswith(".egg-info") for p in path.parts):
            continue
        zf.write(path, path.relative_to(REPO_ROOT))


def package(
    out_path: Path,
    *,
    include_models: bool = True,
    models_dir: Path = DEFAULT_MODELS,
) -> Path:
    if not SRC_DIR.is_dir():
        raise FileNotFoundError(f"Missing source tree: {SRC_DIR}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)

    manifest: dict[str, object] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": None,
        "pipeline": "detector (single-frame 3D U-Net) -> distance-gated bipartite linking",
    }

    # Ship `_best` when it exists -- that is what `get_checkpoint()` prefers, so
    # packaging only the last epoch would silently submit different weights than
    # were validated locally.
    ckpt = models_dir / BEST_CHECKPOINT_NAME
    if not ckpt.exists():
        ckpt = models_dir / CHECKPOINT_NAME
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _add_tree(zf, SRC_DIR)
        if SCRIPTS_DIR.is_dir():
            _add_tree(zf, SCRIPTS_DIR)

        if include_models and ckpt.exists():
            digest = _sha256(ckpt)
            manifest["checkpoint"] = {
                "name": ckpt.name,
                "sha256": digest,
                "bytes": ckpt.stat().st_size,
            }
            try:
                import torch

                meta = torch.load(ckpt, map_location="cpu", weights_only=False)
                manifest["epochs_trained"] = meta.get("epoch")
                manifest["model_config"] = meta.get("config")
                manifest["held_out_volumes"] = meta.get("val_names")
            except Exception as exc:  # noqa: BLE001
                manifest["checkpoint_metadata_error"] = str(exc)
            zf.write(ckpt, Path("models") / ckpt.name)
            print(f"Included {ckpt}  sha256={digest[:16]}...")
        elif include_models:
            print(f"WARNING: no {ckpt} -- packaging code only. Inference will refuse to run.")

        zf.writestr("ARTIFACT_MANIFEST.json", json.dumps(manifest, indent=1))

    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("Upload as a Kaggle Dataset named `cell-tracking-src`, attach it plus the")
    print("competition data, then run notebooks/kaggle_run.ipynb.")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package src/ (+ checkpoint) for Kaggle.")
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--no-models", action="store_true", help="Zip code only.")
    args = parser.parse_args()
    package(args.out, include_models=not args.no_models, models_dir=args.models_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Build and run the reference-pack parity oracle for the 0.942 replication.

Copies ``context/pack_primary/repo`` to a scratch directory, optionally applies
the six inference patches from cell 16 of
``context/biohub-0-942-lb-proxy-score-0-9417.ipynb`` *verbatim* (the notebook's
own ``str.replace`` anchors), sets the notebook's environment variables, and
runs ``scripts/predict_unet_transformer.py`` on the given stems. The resulting
``.geff`` files are the oracle that ``scripts/parity/parity_check.py`` compares
our port against.

Modes:
  vanilla  -- unpatched pack script, single seed, 4-view TTA, edge threshold 0.5
  patched  -- cell-16 patches: D4 TTA, dual seed, retention guard, bidirectional
              harmonic fusion, edge-feature TTA (primary + secondary)

The pack's per-voxel temporal attention flattens B*S > 65535 rows, which
crashes the fused SDPA kernels on L40S/torch 2.6 (see
reports/2026-09-09-sdpa-math-backend-l40s.md), so the script is executed under
the MATH backend.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "context" / "biohub-0-942-lb-proxy-score-0-9417.ipynb"
PACK_PRIMARY = REPO_ROOT / "context" / "pack_primary"
PACK_SECONDARY = REPO_ROOT / "context" / "pack_secondary"

# Cell-8 / cell-14 / cell-16 environment of the committed 0.942 run.
NOTEBOOK_ENV = {
    "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.20",
    "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.80",
    "BIOHUB_SECONDARY_LINK_MODE": "low_margin_consensus",
    "BIOHUB_SECONDARY_MIX_TEMPERATURE": "1",
    "BIOHUB_SECONDARY_LOW_MARGIN_MAX": "0.35",
    "BIOHUB_DUAL_SEED_EDGE_THRESHOLD": "0.48",
    "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION": "0.90",
    "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.15",
    "BIOHUB_BIDIRECTIONAL_FUSION_MODE": "harmonic_probability",
    "BIOHUB_EDGE_FEATURE_TTA": "1",
    "BIOHUB_SECONDARY_EDGE_FEATURE_TTA": "1",
    "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT": "0.75",
    "BIOHUB_DIAGNOSTIC_ARM": "harmonic_association_production",
    "POLARS_PREFER_PKG": "32",
}
ILP_ARGS = [
    "--det-threshold", "0.965",
    "--ilp-edge-weight", "-1.0",
    "--ilp-appearance-weight", "0.0",
    "--ilp-disappearance-weight", "2.0",
    "--ilp-division-weight", "1.2",
    "--use-ilp",
]


def _cell_source(index: int) -> str:
    nb = json.loads(NOTEBOOK.read_text())
    return "".join(nb["cells"][index]["source"])


def apply_cell16_patches(repo_dir: Path, working_dir: Path) -> None:
    """Execute the patch section of notebook cell 16 against ``repo_dir``."""
    src = _cell_source(16)
    start = src.index("# ============================================================\n# 1) Detection TTA")
    end = src.index("# ============================================================\n# 8) Run inference")
    patch_code = src[start:end]
    # The notebook hardcodes /kaggle/working for its diagnostic logs.
    patch_code = patch_code.replace('Path("/kaggle/working")', 'Path(os.environ["BIOHUB_WORKING_DIR"])')
    os.environ["BIOHUB_WORKING_DIR"] = str(working_dir)
    namespace = {
        "__name__": "__cell16__",
        "os": os,
        "json": json,
        "Path": Path,
        "REPO_DIR": repo_dir,
        "WORKING_DIR": working_dir,
    }
    exec(compile(patch_code, "<cell16-patches>", "exec"), namespace)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["vanilla", "patched"], required=True)
    p.add_argument("--stems", nargs="+", required=True, help="dataset stems, e.g. 44b6_1574802b")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "biohub-cell-tracking-during-development" / "train"))
    p.add_argument("--work-dir", required=True, help="scratch dir for the repo copy and outputs")
    p.add_argument("--out-dir", required=True, help="where the oracle .geff files are copied")
    p.add_argument("--unet-batch-size", default="4")
    args = p.parse_args()

    work = Path(args.work_dir).resolve()
    out_dir_abs = Path(args.out_dir).resolve()
    repo_dir = work / f"repo_{args.mode}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    shutil.copytree(PACK_PRIMARY / "repo", repo_dir)
    # weights/ next to config.json, as load_model expects
    (repo_dir / "weights").mkdir(exist_ok=True)
    shutil.copytree(PACK_PRIMARY / "weights" / "unet_transformer", repo_dir / "weights" / "unet_transformer")

    if args.mode == "patched":
        # cell 16's bidirectional patch asserts the weight env var before patching
        os.environ.update(NOTEBOOK_ENV)
        os.environ["BIOHUB_SECONDARY_WEIGHTS"] = str(
            PACK_SECONDARY / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
        )
        apply_cell16_patches(repo_dir, work)

    splits = repo_dir / "oracle_splits.json"
    splits.write_text(json.dumps([{"split": 0, "train": [], "test": list(args.stems)}]))

    sys.path.insert(0, str(repo_dir / "src"))
    sys.path.insert(0, str(repo_dir / "scripts"))
    os.chdir(repo_dir)
    sys.argv = [
        "predict_unet_transformer.py",
        "--data-dir", args.data_dir,
        "--splits", str(splits),
        "--split", "0",
        "--weights", "weights/unet_transformer/split_0/edge_predictor_best.pth",
        "--unet-batch-size", args.unet_batch_size,
        "--method", f"oracle_{args.mode}",
        *ILP_ARGS,
    ]
    print("running:", " ".join(sys.argv), flush=True)

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        runpy.run_path(str(repo_dir / "scripts" / "predict_unet_transformer.py"), run_name="__main__")

    out_dir = out_dir_abs
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = sorted((repo_dir / "predictions").glob(f"*/oracle_{args.mode}/split_0/*.geff"))
    for g in produced:
        dst = out_dir / g.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(g, dst)
    print(f"oracle[{args.mode}] wrote {len(produced)} geff(s) to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

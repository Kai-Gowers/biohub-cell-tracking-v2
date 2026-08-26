#!/usr/bin/env python3
"""Convert predicted `.geff` graphs into `submission.csv`.

    python scripts/make_submission.py --geff-dir dist/preds --out submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cell_tracking.config import get_test_dir
from cell_tracking.submit import write_submission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geff-dir", type=Path, default=Path("dist/preds"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the check that every test dataset has a predicted graph.",
    )
    args = parser.parse_args()

    expected = None
    if not args.no_verify:
        test_dir = args.test_dir or get_test_dir()
        if Path(test_dir).exists():
            expected = sorted(p.stem for p in Path(test_dir).glob("*.zarr"))

    info = write_submission(args.geff_dir, args.out, expected=expected)
    print(f"Wrote {info['path']}: {info['rows']} rows across {info['datasets']} dataset(s)")
    for name, counts in info["per_dataset"].items():
        print(f"  {name}: {counts['nodes']} nodes, {counts['edges']} edges")
    if expected:
        print(f"verified all {len(expected)} test datasets present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

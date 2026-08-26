#!/usr/bin/env python3
"""Selective Kaggle competition data download.

The full dataset is ~90GB. This script only lists files or downloads one file.
There is no default "download everything" path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def list_files() -> None:
    run(["kaggle", "competitions", "files", "-c", COMPETITION])


def download_file(filename: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-f",
            filename,
            "-p",
            str(out_dir),
        ]
    )
    print(f"Downloaded {filename!r} into {out_dir}")
    print("Unzip if needed, e.g.: unzip -d", out_dir, out_dir / Path(filename).name)


def download_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-p",
            str(out_dir),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "List or selectively download Biohub cell-tracking competition files. "
            "Full data is ~90GB — prefer Kaggle notebooks for full runs."
        )
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List competition files (does not download).",
    )
    parser.add_argument(
        "--file",
        metavar="NAME",
        help="Download a single competition file into --out.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=argparse.SUPPRESS,  # hidden; requires confirmation flag
    )
    parser.add_argument(
        "--i-know-this-is-90gb",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if args.all:
        if not args.i_know_this_is_90gb:
            print(
                "Refusing full download (~90GB). "
                "Use --file NAME for a single file, or pass "
                "--all --i-know-this-is-90gb if you really mean it.",
                file=sys.stderr,
            )
            return 2
        download_all(args.out)
        return 0

    if args.list:
        list_files()
        return 0

    if args.file:
        download_file(args.file, args.out)
        return 0

    parser.print_help()
    print(
        "\nExamples:\n"
        "  python scripts/download_data.py --list\n"
        "  python scripts/download_data.py --file sample_submission.csv\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

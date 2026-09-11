#!/usr/bin/env python3
"""Decompose a prediction's edge errors by cause, embryo, and crowding.

    python scripts/analyze_errors.py --geff-dir dist/preds_val_<tag> --held-out <ckpt>
    python scripts/analyze_errors.py --geff-dir ... --held-out ... --candidates dist/preds_val_<tag>/candidates

Same inputs as score_local.py. With a candidate dump (predict.py
--dump-candidates, which slurm_score.sbatch passes by default) it also
reports true-partner rank and greedy-vs-exact regret. See
cell_tracking/analysis.py for the definitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cell_tracking.analysis import VolumeErrors, aggregate, analyze_volume
from cell_tracking.config import get_train_dir
from cell_tracking.io_geff import embryo_of

# Reuse the scorer's checkpoint reader so the held-out list is identical.
from score_local import held_out_names  # noqa: E402  (scripts/ is on sys.path when run as a script)


def _pct(a: float, b: float) -> str:
    return f"{100 * a / b:5.1f}%" if b else "   --"


def print_block(title: str, agg: dict) -> None:
    fn, fp = agg["fn"], agg["fp"]
    print(f"\n=== {title}: {agg['n_volumes']} volume(s) ===")
    print(
        f"det recall {100 * agg['det_recall']:.1f}%   assoc recall (both endpoints detected) "
        f"{100 * agg['assoc_recall']:.1f}%   TP={agg['tp']} FP={fp} FN={fn}"
    )
    print("FN by cause:")
    for k, label in [
        ("fn_undetected_both", "both endpoints undetected"),
        ("fn_undetected_parent", "parent undetected"),
        ("fn_undetected_child", "child undetected"),
        ("fn_wrong_link_both", "both detected, both linked elsewhere"),
        ("fn_wrong_link_parent", "both detected, parent linked to wrong child"),
        ("fn_wrong_link_child", "both detected, child linked to wrong parent"),
        ("fn_unlinked", "both detected, neither linked"),
        ("fn_division_second_child", "division 2nd child (no divisions modelled)"),
    ]:
        print(f"  {label:48s} {agg[k]:6d}  {_pct(agg[k], fn)}")
    det = agg["fn_undetected_both"] + agg["fn_undetected_parent"] + agg["fn_undetected_child"]
    link = agg["fn_wrong_link_both"] + agg["fn_wrong_link_parent"] + agg["fn_wrong_link_child"] + agg["fn_unlinked"]
    print(f"  {'-> detector-attributable':48s} {det:6d}  {_pct(det, fn)}")
    print(f"  {'-> linker-attributable':48s} {link:6d}  {_pct(link, fn)}")
    print("FP by cause:")
    for k, label in [
        ("fp_partner_detected", "true partner detected, linker chose another"),
        ("fp_partner_undetected", "true partner undetected"),
        ("fp_both_matched_wrong_pair", "both endpoints matched, wrong pair"),
    ]:
        print(f"  {label:48s} {agg[k]:6d}  {_pct(agg[k], fp)}")
    if agg.get("rank_hist"):
        tot = sum(agg["rank_hist"].values())
        print("true-partner rank among gated candidates (assoc edges):")
        for k, n in sorted(agg["rank_hist"].items(), key=lambda kv: (int(kv[0]) < 0, int(kv[0]))):
            label = "not a candidate" if k == "-1" else f"rank {k}"
            print(f"  {label:48s} {n:6d}  {_pct(n, tot)}")
    if agg.get("regret_tp") is not None:
        print(
            f"greedy -> exact assignment on the same scores: TP {agg['regret_tp']:+d}, FP {agg['regret_fp']:+d}"
        )
    if agg.get("by_crowding"):
        print("by source-node crowding (nearest neighbour, um):  gt_edges  tp  wrong_link  undetected  assoc_recall")
        for b, d in sorted(agg["by_crowding"].items()):
            both = d["tp"] + d["wrong_link"]
            print(
                f"  {b:14s} {d['gt_edges']:8d} {d['tp']:5d} {d['wrong_link']:11d} {d['undetected']:11d}  "
                f"{_pct(d['tp'], both)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geff-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--held-out", type=Path, required=True, help="Checkpoint whose val_names to analyze.")
    parser.add_argument("--candidates", type=Path, default=None, help="Dir of <name>.npz candidate dumps.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    train_dir = args.train_dir or get_train_dir()
    names, val_prefix = held_out_names(args.held_out)
    names = [n for n in names if (args.geff_dir / f"{n}.geff").exists()]
    cand_dir = args.candidates
    if cand_dir is None and (args.geff_dir / "candidates").exists():
        cand_dir = args.geff_dir / "candidates"

    results: list[VolumeErrors] = [analyze_volume(n, train_dir, args.geff_dir, cand_dir) for n in names]
    print_block("ALL held-out" + (f" (embryo {val_prefix})" if val_prefix else ""), aggregate(results))
    by_embryo: dict[str, list[VolumeErrors]] = {}
    for r in results:
        by_embryo.setdefault(embryo_of(r.name), []).append(r)
    if len(by_embryo) > 1:
        for emb, rs in sorted(by_embryo.items()):
            print_block(f"embryo {emb}", aggregate(rs))
    if cand_dir is None:
        print("\n(no candidate dump found: rank and regret not computed; re-predict with --dump-candidates)")
    if args.json_out:
        args.json_out.write_text(
            json.dumps({"volumes": [r.to_dict() for r in results], "all": aggregate(results)}, indent=1)
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

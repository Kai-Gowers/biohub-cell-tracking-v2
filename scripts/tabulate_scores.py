"""Tabulate dist/score_<tag>.json files (score_local.py --json-out) with per-embryo subtotals.

    python scripts/tabulate_scores.py ddp_best_blend ddp_best_s314159 ... [--md]
"""
import argparse, json, sys
from pathlib import Path

def agg(rows):
    w = sum(r["weight"] for r in rows)
    adj = sum(r["adjusted_edge_jaccard"] * r["weight"] for r in rows) / w
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    dtp, dfp, dfn = (sum(r[k] for r in rows) for k in ("div_tp", "div_fp", "div_fn"))
    div = dtp / (dtp + dfp + dfn) if (dtp + dfp + dfn) else 0.0
    det = sum(r["det_within_7um"] for r in rows) / sum(r["n_gt_nodes"] for r in rows)
    return dict(score=adj + 0.1 * div, adj=adj, div=div, det=det, tp=tp, fp=fp, fn=fn,
                dtp=dtp, dfp=dfp, dfn=dfn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    sep = " | " if a.md else "  "
    hdr = ["tag", "SCORE", "44b6", "6bba", "adj", "div(tp/fp/fn)", "det%", "FP", "FN"]
    print(sep.join(hdr))
    if a.md:
        print("|".join(["---"] * len(hdr)))
    for tag in a.tags:
        p = Path(f"dist/score_{tag}.json")
        if not p.exists():
            print(f"{tag}: missing"); continue
        rows = json.loads(p.read_text())
        t = agg(rows)
        by = {e: agg([r for r in rows if r["embryo"] == e]) for e in sorted({r["embryo"] for r in rows})}
        print(sep.join([tag, f"{t['score']:.4f}",
                        f"{by.get('44b6', {}).get('score', float('nan')):.4f}",
                        f"{by.get('6bba', {}).get('score', float('nan')):.4f}",
                        f"{t['adj']:.4f}", f"{t['div']:.3f} ({t['dtp']}/{t['dfp']}/{t['dfn']})",
                        f"{100*t['det']:.1f}", str(t["fp"]), str(t["fn"])]))

if __name__ == "__main__":
    main()

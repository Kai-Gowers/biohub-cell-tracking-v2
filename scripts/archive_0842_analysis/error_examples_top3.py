#!/usr/bin/env python3
"""Three single-row figures for the PI: one typical example per top error, plain-language caption."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import error_examples as EE

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
M = pd.read_csv(out / "misses.csv"); W = pd.read_csv(out / "linker_fn.csv")
plt.rcParams.update({"font.size": 9})

def row_fig(kind, rec, caption, fname):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.05]))
    sc = EE.Scene(rec["name"])
    (EE.draw_detector_row if kind == "det" else EE.draw_linker_row)(axes, sc, rec)
    fig.suptitle(caption, fontsize=10.5, y=0.99, ha="left", x=0.02, va="top", linespacing=1.4)
    fig.subplots_adjust(top=0.70, bottom=0.02, left=0.02, right=0.99, wspace=0.06)
    fig.savefig(out / fname); plt.close(fig)
    print("wrote", fname)

m = M[(M.name == "44b6_d5e7d891") & (M.t == 16) & M.mech.str.startswith("D3")].iloc[0]
row_fig("det", m,
    "TOP ERROR 1  --  the detector sees the cell but hands its peak to a neighbour   (65% of all missed cells, both embryos)\n"
    "The detector's probability at the ringed cell is 0.999 (threshold 0.985), yet no detection is placed there: the response keeps rising toward the\n"
    "neighbouring nucleus 8 µm away, so the peak-finder keeps only that neighbour's peak. The cell is clearly visible and clearly separate in the image.\n"
    "green o = annotated cell, red + = detection; white ring = the missed cell, red ring = the detection nearest to it.  Left three: xy at t-1, t, t+1. Right: side view (xz).",
    "top1_detector_peak_absorbed_by_neighbour.png")

w = W[(W.name == "44b6_d5e7d891") & (W.t == 18) & W.mech.str.startswith("L1a")].iloc[0]
row_fig("link", w,
    "TOP ERROR 2  --  the detector places the same cell at a different depth (z) in consecutive frames, and the linker follows the wrong cell   (~60% of linking errors)\n"
    "The annotated cell moved 2 µm (green arrow). Its two detections, however, sit 6 µm apart in z (side views, right), because the detector's z estimate jittered by 3-4 slices.\n"
    "Fed a 6 µm jump, the pair scorer prefers the neighbour that stayed at the source's depth (yellow ring, score 1.00 vs 1.00 for the true partner).\n"
    "white ring = the cell (t) / its true partner (t+1), red ring = their detections, yellow = the detection the linker chose; red x = where the cell's detection was in frame t.",
    "top2_linker_z_jitter.png")

m = M[(M.name == "6bba_3a1849c2") & (M.t == 37) & M.mech.str.startswith("D1")].iloc[0]
row_fig("det", m,
    "TOP ERROR 3  --  dim cells the detector never sees   (21% of missed cells; all in embryo 6bba, mostly three videos; lost for ~11 frames at a time)\n"
    "The ringed cell is 1.7x the frame's median intensity (typical cells: 6x). The detector's probability there peaks at 0.75, under the 0.985 threshold, for 32 consecutive frames.\n"
    "Nothing is detected within 10 µm in any frame. These are faint nuclei deep in the tissue, an imaging limit as much as a model limit.\n"
    "green o = annotated cell, red + = detection; white ring = the missed cell.  Left three: xy at t-1, t, t+1. Right: side view (xz).",
    "top3_detector_dim_cell.png")

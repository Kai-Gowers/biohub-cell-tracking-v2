# Archived: error analysis of the retired 0.842 pipeline

These scripts produced the figures and counts in

- `reports/2026-09-14-error-analysis-0842.md` (+ `-detailed-appendix.md`) and `reports/figures/2026-09-14-error-analysis-0842/`
- `reports/2026-09-16-error-categories-with-examples.md`, `reports/2026-09-16-top3-errors-for-pi.md` and `reports/figures/2026-09-16-error-examples/`
- `reports/2026-09-15-error-analysis-summary.md`

They import the repo's **previous** architecture (2-frame U-Net detector + pair MLP + greedy linker,
`main` at commit 292a8c3, leaderboard 0.842), which was replaced on 2026-09-11 by the port of the public
0.942 pipeline that now lives in `src/`. They do **not** run against the current `src/cell_tracking`.

To rerun them, check out the old tree next to this repo and run from there:

```bash
git worktree add /projects/twist2d/gowers/biohub-cell-tracking-v2-main 292a8c3
cp scripts/archive_0842_analysis/*.py /projects/twist2d/gowers/biohub-cell-tracking-v2-main/scripts/
# then follow the "how to reproduce" notes in the 2026-09-14 report (section 9 for the 3D error browser)
```

| script | what it did |
|---|---|
| `dataset_summary.py` | plain summary of the competition data (annotated vs estimated cells per frame, per embryo) |
| `error_report.py` | full error analysis of one held-out prediction: statistics, figures, qualitative panels |
| `error_browser.py` | interactive 3D browser over every counted tracking error (`--port 8765`) |
| `error_examples_build.py` | per-miss / per-wrong-link tables (pickled) used by `error_examples.py` |
| `error_examples_probe*.py` | re-run the old detector at missed GT nodes to classify misses (peak absorbed by neighbour, dim cell, ...) |
| `error_examples.py` | frequency-ranked error categories with typical xy / xz examples (fig13, fig14) |
| `error_examples_top3.py` | the three single-row figures for the PI |

The *findings* (65% of misses are a peak absorbed by a neighbour, z-jitter drives linker errors, crowding explains
most of the 44b6 vs 6bba gap) carry over to the current pipeline; the code does not.

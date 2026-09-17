### Detector: 685 missed annotated cells (of 14177; 44b6 101, 6bba 584)

| mech | n | pct | n44 | snr_med | dim_pct | bright_pct | near_med | run_med | p_max2_med |
|---|---|---|---|---|---|---|---|---|---|
| D1 sustained miss (>=3 frames), field below threshold | 134 | 19.56 | 0 | 1.73 | 58.96 | 0.00 | 10.44 | 11.00 | 0.96 |
| D2 transient miss (1-2 frames), field below threshold | 12 | 1.75 | 0 | 1.93 | 50.00 | 0.00 | 8.55 | 1.00 | 0.97 |
| D3 field above threshold but no peak kept (ridge into a neighbour) | 448 | 65.40 | 74 | 3.94 | 21.88 | 15.18 | 8.40 | 2.00 | 1.00 |
| D4 merged: nearest detection claimed by a neighbouring GT cell | 6 | 0.88 | 6 | 1.22 | 100.00 | 0.00 | 5.83 | 7.00 | 1.00 |
| D5 local-metric truncation artefact (detection 5.6-7 um away, mostly in z) | 85 | 12.41 | 21 | 2.59 | 34.12 | 16.47 | 6.68 | 2.00 | 1.00 |

run length x brightness (all misses):
| run_cls | <2 dim | 2-4 | 4-8 | >8 bright | All |
|---|---|---|---|---|---|
| 1 frame | 58 | 80 | 60 | 27 | 225 |
| 2 frames | 22 | 35 | 38 | 27 | 122 |
| >=3 frames | 138 | 94 | 78 | 28 | 338 |
| All | 218 | 209 | 176 | 82 | 685 |

### Linker: 708 counted FN edges with both endpoints detected

| mech | n | pct | n44 | s_true_med | s_chosen_med | crowd_med | gt_disp_med | det_dz_gt3_pct |
|---|---|---|---|---|---|---|---|---|
| L1a wrong partner chosen, near-tie (margin < 0.1) | 150 | 21.19 | 51 | 0.98 | 0.99 | 8.95 | 2.07 | 69.33 |
| L1b wrong partner chosen, confident (margin > 0.5) | 130 | 18.36 | 34 | 0.11 | 0.99 | 8.89 | 2.07 | 79.23 |
| L1b wrong partner chosen, intermediate margin | 79 | 11.16 | 22 | 0.74 | 0.99 | 8.85 | 2.19 | 64.56 |
| L1c wrong partner: true partner ranked first but taken (greedy) | 23 | 3.25 | 5 | 0.99 | 0.93 | 6.62 | 2.87 | 65.22 |
| L1d wrong partner: true partner beyond the 15 um gate | 4 | 0.56 | 1 | nan | 0.97 | 10.77 | 15.30 | 75.00 |
| L2 true target stolen by an unannotated competitor; cell left unlinked | 207 | 29.24 | 55 | 0.86 | nan | 7.33 | 1.99 | 68.60 |
| L3 true pair scored below 0.5; cell left unlinked | 97 | 13.70 | 5 | 0.19 | nan | 9.87 | 1.72 | 36.08 |
| L4 division (second child never linked) | 18 | 2.54 | 3 | 0.90 | 0.99 | 10.94 | 6.68 | 44.44 |

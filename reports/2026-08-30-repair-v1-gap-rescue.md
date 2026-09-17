# Repair v1: dangling-link rescue only

## Why this shape, not "close a one-frame gap"

The classic repair operator from the sibling repo (`gap1`/`gap2`: bridge a
track across 1-2 missed frames) doesn't transfer here, for a reason specific
to this repo's format, not a design preference: the competition's ground
truth only ever contains **dt=1** edges, so a dt=2 bridge edge is guaranteed
not to score a match, and `TrackGraph.validate()` (`graph.py`) hard-asserts
dt=1 on every edge -- a gap-bridging edge would fail validation outright.
Closing an actual frame gap without violating that would require inventing a
node at the skipped frame, which is a materially bigger and less
conservative change (fabricating geometry) than "start small, obvious
mistakes only" calls for.

What's left within dt=1-only, no-new-nodes: a node's true partner in the
*very next* frame was detected, but either (a) sat just outside
`LINK_RADIUS_UM` (15um), or (b) lost a shared candidate to some other node
during `link.py`'s greedy score-sorted pass. Both are genuinely "obvious" in
the sense that the correct partner node already exists in the graph --
nothing is fabricated, just a same-frame-pair edge the primary pass missed.

Note: case (b) specifically can *only* happen when the winning competitor
scored higher in the greedy ordering -- a genuinely contested/ambiguous
assignment, not a clear mistake. This pass does not attempt to resolve those
(no re-ranking, no undoing an existing edge) -- it only touches nodes that
are *still completely dangling* after the primary pass, which mostly
isolates case (a): a real partner that simply exceeded the radius.

## What it does

`repair.rescue_dangling_links` (new module): for every consecutive frame
pair, take the nodes still fully dangling after linking (out-degree 0 at t,
in-degree 0 at t+1) and re-run the *same* `link_frames` greedy matcher on
just that residual, free set, at a wider radius (`GAP_RESCUE_RADIUS_UM =
20.0` vs. the primary pass's 15.0 -- a small, deliberately conservative
first step, not swept yet). No new nodes, no non-dt=1 edges, no change to
anything the primary pass already linked. `TrackGraph.validate()` still
holds afterward (re-asserted in `predict_volume` when `repair=True`).

Off by default (`repair=False` / no `--repair` flag) everywhere.

## Held-out result: net harmful, not just inert (2026-08-30)

`detector_best_0841.pt`, 20 held-out volumes, `--repair` at the default
`GAP_RESCUE_RADIUS_UM=20.0` vs. off:

| | TP | FP | FN | edge_jaccard | adjusted | SCORE |
|---|---|---|---|---|---|---|
| repair off | 11867 | 1036 | 1825 | 0.8057 | 0.8031 | **0.8031** |
| repair on | 11890 | 1441 | 1802 | 0.7857 | 0.7831 | **0.7831** |
| Δ | +23 | **+405** | −23 | | | **−0.0200** |

Rescue fired a lot (600-1000+ extra edges per volume, e.g. 6bba_c328f2fd
28,408 -> 29,432), and it does do what it says: FN dropped and TP rose. But
the FP increase is 17.6x the TP gain -- for every real missed link the wider
radius recovers, it wrongly connects ~17-18 pairs of nodes that shouldn't be
linked. Node counts here are ~2 orders of magnitude denser than annotated GT
(node ratio 0.994x is against the *estimated true cell count*, not the
sparse annotation -- most detections are real-but-unannotated or duplicate
cells packed close together). Widening the gate from 15um to 20um mostly
buys wrong connections among that dense unannotated pool, not genuine
rescues of an isolated missed link -- the "obvious mistake" framing assumed
a sparser neighborhood than this data actually has.

**Verdict: do not ship this at radius=20. `repair=False` stays the default.**
This is a real, above-noise result (−0.02, both runs 20/20 volumes, same
checkpoint/settings otherwise), not measurement noise.

## v2: density-adaptive gating (2026-08-30, same day)

Per `reports/2026-08-30-933-notebook-analysis.md` finding #1 (a public 0.933
notebook's gap-closer sizes its gate from local neighbor spacing instead of a
flat radius): `rescue_dangling_links` now computes each node's local
3-nearest-neighbor spacing (`GAP_DENSITY_NEIGHBORS`, median distance, over
every detection in the frame, not just dangling ones) and only *widens*
`GAP_RESCUE_BASE_RADIUS_UM` (now `LINK_RADIUS_UM`, 15um, not v1's flat 20um)
above a reference spacing (`GAP_DENSITY_REFERENCE_UM=10.5`, the empirical
median 3-NN spacing across our own held-out predictions), clamped to
`GAP_DENSITY_MAX_WIDEN_UM=5.0`. One-sided by design: never narrower than
base, so dense neighborhoods never get relaxed past the primary pass's own
radius.

Same held-out setup (`detector_best_0841.pt`, 20 volumes):

| | TP | FP | FN | SCORE |
|---|---|---|---|---|
| repair off | 11867 | 1036 | 1825 | 0.8031 |
| v1 (flat 20um) | 11890 | 1441 | 1802 | 0.7831 |
| v2 (density-adaptive) | 11890 | 1301 | 1802 | **0.7904** |

Improved but still net harmful. Identical TP to v1 (every real gap-rescue v1
found, v2 finds too -- they all happen to sit in locally sparse
neighborhoods), and FP dropped by 140 (1441 -> 1301, about a third of v1's
excess FP) with zero TP cost -- density gating is doing exactly what it was
supposed to, suppressing wrong connections in crowded regions without losing
real ones. But the remaining spurious:real ratio is still ~11.5:1 (down from
17.6:1). Geometry alone, even well-gated, isn't precise enough on this data.

**Verdict: still don't ship. `repair=False` stays the default.** The
directionally-correct fix (finding #1) measurably helped; it just wasn't
enough alone. Finding #4 from the same notebook analysis -- fold the
existing trained edge scorer's probability into the rescue cost, not
distance/density alone -- is the natural next thing to try if this is
revisited, since the failure mode that remains (wrong-but-close pairs in
dense regions) is exactly what a learned signal, not geometry, should be
better at discriminating.

## How to apply

This is v1/v2 -- one operator, incrementally refined, twice measured, still
losing both times. The dense/unannotated-node reality of this data means
"still dangling after the primary pass" is not itself a strong enough
"obvious mistake" signal -- most dangling nodes are dangling because there
is no real partner nearby, not because a real partner was barely missed, and
geometry (flat or density-adaptive) can only partially tell those apart.
Don't re-try a pure-geometry variant (a different reference spacing, a
different gain/clamp) expecting it to flip the sign -- v1 -> v2 already
spent that lever and it closed less than half the gap. The next thing worth
trying, if this is revisited, is folding the existing trained edge scorer's
probability into the rescue cost alongside distance/density (notebook
finding #4) -- untried so far. Do not add gap1/gap2-style multi-frame
bridging or any node-inventing repair without first checking compatibility
with `TrackGraph.validate()`'s dt=1 requirement (it isn't compatible,
as-is).

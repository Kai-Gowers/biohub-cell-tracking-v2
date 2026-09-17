# Analysis: second downloaded notebook (biohub-c30-public-0-930-v18-exact.ipynb)

Read-only analysis, code never copied. Follows
`reports/2026-08-30-933-notebook-analysis.md` (same session, first
notebook).

## Important correction to the premise: this is the same pipeline, not an independent one

Diffed this notebook cell-by-cell against the first one analyzed
(`933-biohub-bidir30.ipynb`). Across the two ~1700-line core-logic cells
(dependency/patch application, and the full post-processing module), the
literal diff is **two numeric constants**:

| constant | this notebook (LB 0.930) | first notebook (LB 0.933) |
|---|---|---|
| `BIOHUB_SECONDARY_DETECTION_WEIGHT` | 0.70 | 0.80 |
| `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT` | 0.15 | 0.30 |

Everything else -- every function, every other threshold, the config-guard
print strings, even a leftover hardcoded `"Reverse-time association weight:
0.200"` log line that neither version's guard actually reflects -- is
byte-identical. **These are two saved versions of the same author/team's
sweep, not two independently-arrived-at "top notebooks."** Worth flagging
because the question "why is X so prominent across the top notebooks" only
holds if the notebooks are actually independent; here it isn't -- one data
point wearing two filenames. Treat conclusions below as one team's sweep,
not a cross-team consensus, until a genuinely different lineage is checked.

## Does that still answer "is harmonic fusion a big deal?" -- yes, with numbers now

Both notebooks state their common ancestor is a "fixed-90 dual-seed clean
pipeline" scoring LB 0.913 with bidirectional fusion off. From there:

- bidir=0.15, secondary-detection=0.70 -> **0.930** (+0.017 over the 0.913 ancestor)
- bidir=0.30, secondary-detection=0.80 -> **0.933** (+0.003 more)

Both parameters move together at each step, so this doesn't cleanly isolate
harmonic fusion from the secondary-model blend weight -- but the combined
"add reverse-time harmonic fusion + lean more on the second model" change is
worth about +0.02 to this team, with the first increment (0->0.15/0.475
->0.70) mattering far more than the second (0.15->0.30, 0.70->0.80,
diminishing returns). That's consistent with turning a real, useful signal
on being worth more than fine-tuning how hard to lean on it -- an argument
for trying bidirectional fusion (finding #6 from the first report) before
spending time tuning its exact weight.

## Node transformer + ILP: not new information, same architecture as before

Confirmed identical model family (`unet_transformer`) and identical
`tracksdata`+ILP linking as the first notebook -- nothing new to report
here since it's the same pipeline. Their prominence across "top notebooks"
generally is plausible on priors (this competition's public leaderboard is
known to have converged on a small number of shared architectural patterns)
but isn't demonstrated by comparing these two specific files, since they're
the same one.

## What's genuinely new here: an in-repo before/after for safe division

This notebook's post-processing cell carries a comment the first notebook's
didn't -- a change explicitly credited to a different competitor's public
repo (`kunaldesale2408/biohub-cell-tracking`), with a concrete before/after:

> Our rule already used orphans, the same `d(parent,candidate) + 0.15 *
> d(sister,candidate)` score and the same rate caps, but had NO structural
> constraints -- which is why it produced hundreds of forks and never a
> single division true positive.

Three constraints were added, and this is now a *second*, differently-sourced
confirmation of `2026-08-30-933-notebook-analysis.md` finding #3
(mutual-nearest-orphan + forward divergence turn division recovery from
broken to working) -- plus one gate the first notebook's version didn't
make explicit:

1. **The parent must already be mid-track** (has its own predecessor, not a
   track start) before it's considered eligible to propose a division at
   all. A node with no track history yet has no established single-child
   pattern or motion context to test a division proposal against.
2. Mutual-nearest-orphan (same as before).
3. Forward divergence at t+2 (same as before).

This "hundreds of forks, zero true positives" data point is exactly the
failure mode CLAUDE.md's history warns about for this repo's own sibling
project (naive division recovery measuring inert) and for our own repair v1
(a single geometric threshold, no structural constraints, net harmful) --
now confirmed a third time, from a third independent codebase. The
structural constraints, not the distance thresholds, are what make the
difference between "generates garbage" and "recovers real divisions."

## A cross-check worth knowing about, not acting on

This notebook prints its own held-out number explicitly as a **"proxy
score"**: `PROXY_SCORE 0.9438` against an actual leaderboard score of
0.930 -- a ~0.014 optimistic gap between their local validation and the
real submission. That's the same shape of caution this repo's own
`metric.py` docstring and `score_local.py` already carry (trust
same-checkpoint comparisons, distrust cross-checkpoint/cross-split ones).
Good independent confirmation that skepticism is warranted industry-wide,
not just a quirk of this repo's own history -- not a new action item.

## Net effect on recommendations

No new repair operator to add to the ranked list from the first report.
This strengthens finding #3 (safe division's structural gates) with a
second data point and adds one missing gate (parent-must-be-mid-track), and
gives finding #6 (bidirectional fusion) a rough magnitude (+0.02 combined
with secondary-model weight, front-loaded in the first increment) instead
of just a mechanism description. Otherwise: same architecture, same
conclusions, corrected premise that these are sibling configs of one
pipeline rather than two separate competitors.

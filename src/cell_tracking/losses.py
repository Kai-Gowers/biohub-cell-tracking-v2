"""Detection loss, plus the edge-scorer loss for the learned linking stage."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cell_tracking.config import CENTER_NEG_ALPHA


def detection_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = CENTER_NEG_ALPHA,
) -> torch.Tensor:
    """Weighted BCE on logits with per-sample positive/negative normalization.

        w+(b) = 1 / N+(b),   w-(b) = alpha / N-(b),   alpha = 0.01

    `target` is BINARY -- ground-truth node voxels are 1, every other voxel
    is a weak negative. Not a Gaussian heatmap: only ~2.8% of real cells are
    annotated, so the negative pool is mostly unlabelled true cells, and
    alpha is what keeps them from dominating. Total positive weight is 1.0
    per sample against 0.01 total negative weight -- a deliberate 100:1 bias
    toward recall, which is what makes a 0.985 detection threshold meaningful
    rather than an arbitrary tail cut.

    Samples with no positives contribute their negative term only (the
    positive term is dropped rather than dividing by zero), so frames with no
    annotation in them still teach the model what background looks like.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    b = logits.shape[0]
    flat_bce = bce.reshape(b, -1)
    flat_pos = (target.reshape(b, -1) > 0.5).to(logits.dtype)
    flat_neg = 1.0 - flat_pos

    n_pos = flat_pos.sum(dim=1)
    n_neg = flat_neg.sum(dim=1).clamp_min(1.0)

    pos_term = torch.where(
        n_pos > 0,
        (flat_pos * flat_bce).sum(dim=1) / n_pos.clamp_min(1.0),
        torch.zeros_like(n_pos),
    )
    neg_term = alpha * (flat_neg * flat_bce).sum(dim=1) / n_neg
    return (pos_term + neg_term).mean()


def parent_softmax(logits: torch.Tensor, dst_idx: torch.Tensor, n_dst: int) -> torch.Tensor:
    """P_ij = softmax_i(l_ij): normalize each candidate edge against the other
    candidates competing for the same target (destination) node, instead of
    scoring every candidate independently.

    `logits` is a flat list of candidate pairs; `dst_idx[p]` names which
    target node candidate pair `p` competes for, so this is a segmented
    softmax, grouped by target rather than by source -- normalizing over
    parents (not children) is what would let one parent connect to two
    daughters without stealing probability from its sibling, should
    divisions ever be added; this baseline still enforces in-degree <= 1 via
    `link.py`'s greedy selection regardless.

    Adapted from the sibling repo's `losses.parent_softmax`
    (`../biohub-cell-tracking`), which measured *removing* this
    normalization (in favor of independent per-pair scoring) as losing 70%
    of true edges vs. 3% for the normalized version -- see
    reports/2026-08-28-parent-softmax-edge-loss.md.

    Computed in fp32 regardless of the incoming dtype: under AMP the logits
    arrive as fp16, whose smallest normal value (~6e-5) would make the
    1e-12 denominator guard round to zero and return infinities. There are
    only a few thousand candidate pairs per step, so the upcast is free.
    """
    if logits.numel() == 0:
        return logits
    logits = logits.float()
    max_per_dst = logits.new_full((n_dst,), float("-inf")).scatter_reduce(
        0, dst_idx, logits, reduce="amax", include_self=True
    )
    shifted = torch.exp(logits - max_per_dst[dst_idx])
    denom = logits.new_zeros(n_dst).scatter_add(0, dst_idx, shifted)
    return shifted / denom[dst_idx].clamp_min(1e-12)


def edge_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    dst_idx: torch.Tensor,
    n_dst: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BCE over parent-softmax-normalized probabilities.

    Candidates are gated to `LINK_RADIUS_UM` and grouped by target (dst)
    node; `parent_softmax` makes each target's candidates compete for its
    probability mass instead of being scored independently.

    A target with no true parent among its candidates -- the real parent
    wasn't detected this frame, or the target genuinely starts a new track
    -- can't be represented by a softmax over only the candidates present:
    probabilities that must sum to 1 can't also all be pushed toward 0.
    That isn't a hard example, it's a contradiction, so such targets are
    masked out of the loss entirely rather than trained against.

    Returns `(loss, probs)` so callers (train diagnostics, prediction) can
    reuse the normalized probabilities instead of recomputing them.
    """
    probs = parent_softmax(logits, dst_idx, n_dst)
    if logits.numel() == 0:
        return logits.new_zeros(()), probs
    has_positive = torch.zeros(n_dst, dtype=torch.bool, device=logits.device)
    pos_dst = dst_idx[labels > 0.5]
    if pos_dst.numel():
        has_positive[pos_dst] = True
    mask = has_positive[dst_idx]
    if not bool(mask.any()):
        return logits.new_zeros(()), probs
    p = probs.clamp(1e-6, 1.0 - 1e-6)
    bce = -(labels * torch.log(p) + (1.0 - labels) * torch.log(1.0 - p))
    return bce[mask].mean(), probs

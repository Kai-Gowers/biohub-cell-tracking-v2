"""Detection loss, plus the edge-scorer loss for the learned linking stage."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cell_tracking.config import CENTER_NEG_ALPHA, EDGE_NEG_ALPHA


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


def edge_loss(logits: torch.Tensor, labels: torch.Tensor, *, alpha: float = EDGE_NEG_ALPHA) -> torch.Tensor:
    """Weighted BCE over candidate edges, same shape of reasoning as `detection_loss`.

    Candidates are gated to `LINK_RADIUS_UM` and labeled via detect-and-match
    (`edge_train.py`), so the overwhelming majority are non-edges -- either a
    genuine distractor or one/both endpoints unmatched to any annotated GT
    node. `alpha` down-weights that negative pool the same way
    `CENTER_NEG_ALPHA` does for background voxels.
    """
    if logits.numel() == 0:
        return logits.new_zeros(())
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    pos = (labels > 0.5).to(logits.dtype)
    neg = 1.0 - pos
    n_pos, n_neg = pos.sum(), neg.sum().clamp_min(1.0)
    pos_term = (pos * bce).sum() / n_pos.clamp_min(1.0) if n_pos > 0 else logits.new_zeros(())
    neg_term = alpha * (neg * bce).sum() / n_neg
    return pos_term + neg_term

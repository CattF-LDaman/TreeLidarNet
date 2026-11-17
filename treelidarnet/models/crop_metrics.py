"""Binary classification metrics for crop verifier validation."""

from __future__ import annotations

import torch


def crop_val_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """probs, labels: 1D same length."""
    labels = labels.detach().float().cpu()
    probs = probs.detach().float().cpu()
    if labels.numel() == 0:
        return {}

    pos = labels >= 0.5
    neg = ~pos
    pred_pos = probs >= threshold

    tp = ((pred_pos) & pos).sum().item()
    fp = ((pred_pos) & neg).sum().item()
    fn = ((~pred_pos) & pos).sum().item()
    tn = ((~pred_pos) & neg).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    out: dict[str, float] = {
        "accuracy": (tp + tn) / labels.numel(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_prob_pos": float(probs[pos].mean()) if pos.any() else float("nan"),
        "mean_prob_neg": float(probs[neg].mean()) if neg.any() else float("nan"),
    }
    if pos.any() and neg.any():
        import numpy as np

        out["auroc"] = _auroc(probs.numpy(), (labels.numpy() >= 0.5).astype(np.int8))
    return out


def _auroc(scores, labels) -> float:
    """Rank-based AUROC (no sklearn)."""
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
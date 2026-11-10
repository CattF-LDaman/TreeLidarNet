"""Focal loss for heatmap training."""

from __future__ import annotations

import torch


def focal_loss_heatmap(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore: torch.Tensor | None = None,
    unknown: torch.Tensor | None = None,
    hard_negative: torch.Tensor | None = None,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    neg_beta_hard: float = 4.0,
) -> torch.Tensor:
    """CenterNet-style focal loss; unknown mask skips negatives (canopy v2)."""
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos = target >= 0.99
    neg = ~pos
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos
    neg_weight = torch.pow(1 - target, beta)
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weight * neg
    if unknown is not None:
        allow_neg = (1.0 - unknown).clamp(0, 1)
        neg_loss = neg_loss * allow_neg
    if hard_negative is not None:
        hn = hard_negative.clamp(0, 1)
        neg_hard = -torch.log(1 - pred) * torch.pow(pred, alpha) * hn
        neg_weight_h = torch.pow(1 - target, neg_beta_hard)
        neg_loss = neg_loss + neg_hard * neg_weight_h
    loss = pos_loss + neg_loss
    if ignore is not None:
        mask = ignore < 1
        loss = loss * mask.float()
        denom = mask.float().sum().clamp(min=1.0)
    else:
        denom = torch.tensor(loss.numel(), device=loss.device, dtype=loss.dtype)
    return loss.sum() / denom


def focal_loss_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore: torch.Tensor | None = None,
    unknown: torch.Tensor | None = None,
    hard_negative: torch.Tensor | None = None,
    **kwargs: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, pos_mean, neg_mean for TensorBoard."""
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos = target >= 0.99
    neg = ~pos
    alpha = float(kwargs.get("alpha", 2.0))
    beta = float(kwargs.get("beta", 4.0))
    neg_beta_hard = float(kwargs.get("neg_beta_hard", 4.0))
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos
    neg_weight = torch.pow(1 - target, beta)
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weight * neg
    if unknown is not None:
        neg_loss = neg_loss * (1.0 - unknown).clamp(0, 1)
    if hard_negative is not None:
        hn = hard_negative.clamp(0, 1)
        neg_loss = neg_loss + (
            -torch.log(1 - pred)
            * torch.pow(pred, alpha)
            * hn
            * torch.pow(1 - target, neg_beta_hard)
        )
    if ignore is not None:
        mask = ignore < 1
        pos_loss = pos_loss * mask.float()
        neg_loss = neg_loss * mask.float()
    total = pos_loss + neg_loss
    n_pos = pos.float().sum().clamp(min=1.0)
    n_neg = neg.float().sum().clamp(min=1.0)
    if ignore is not None:
        denom = mask.float().sum().clamp(min=1.0)
        total_mean = total.sum() / denom
    else:
        total_mean = total.mean()
    return total_mean, pos_loss.sum() / n_pos, neg_loss.sum() / n_neg

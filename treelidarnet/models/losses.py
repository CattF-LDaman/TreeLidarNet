"""Focal loss for heatmap training."""

from __future__ import annotations

import torch


def focal_loss_heatmap(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore: torch.Tensor | None = None,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Modified focal loss (CenterNet-style) on sigmoid pred."""
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos = target >= 0.99
    neg = ~pos
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos
    neg_weight = torch.pow(1 - target, beta)
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weight * neg
    loss = pos_loss + neg_loss
    if ignore is not None:
        mask = ignore < 1
        loss = loss * mask.float()
        denom = mask.float().sum().clamp(min=1.0)
    else:
        denom = torch.tensor(loss.numel(), device=loss.device, dtype=loss.dtype)
    return loss.sum() / denom

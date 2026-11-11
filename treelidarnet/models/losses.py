"""Focal loss for heatmap training."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F

# fp16 cannot represent 1 - 1e-4 (it rounds to 1). A saturated sigmoid then
# makes -log(1 - pred) NaN. Loss math always runs in fp32.
_PROB_EPS = 1e-4


@contextmanager
def _fp32_loss(pred: torch.Tensor) -> Iterator[None]:
    device_type = pred.device.type if pred.device.type in {"cuda", "cpu"} else "cpu"
    with torch.amp.autocast(device_type, enabled=False):
        yield


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
    with _fp32_loss(pred):
        return _focal_loss_heatmap(
            pred.float(),
            target.float(),
            None if ignore is None else ignore.float(),
            None if unknown is None else unknown.float(),
            None if hard_negative is None else hard_negative.float(),
            alpha=alpha,
            beta=beta,
            neg_beta_hard=neg_beta_hard,
        )


def _focal_loss_heatmap(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore: torch.Tensor | None,
    unknown: torch.Tensor | None,
    hard_negative: torch.Tensor | None,
    *,
    alpha: float,
    beta: float,
    neg_beta_hard: float,
) -> torch.Tensor:
    pred = pred.clamp(_PROB_EPS, 1.0 - _PROB_EPS)
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
    with _fp32_loss(pred):
        return _focal_loss_components(
            pred.float(),
            target.float(),
            None if ignore is None else ignore.float(),
            None if unknown is None else unknown.float(),
            None if hard_negative is None else hard_negative.float(),
            alpha=float(kwargs.get("alpha", 2.0)),
            beta=float(kwargs.get("beta", 4.0)),
            neg_beta_hard=float(kwargs.get("neg_beta_hard", 4.0)),
        )


def _focal_loss_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore: torch.Tensor | None,
    unknown: torch.Tensor | None,
    hard_negative: torch.Tensor | None,
    *,
    alpha: float,
    beta: float,
    neg_beta_hard: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred.clamp(_PROB_EPS, 1.0 - _PROB_EPS)
    pos = target >= 0.99
    neg = ~pos
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


def per_tree_peak_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore: torch.Tensor | None = None,
    *,
    alpha: float = 2.0,
    radius_px: int = 8,
    min_target: float = 0.99,
) -> torch.Tensor:
    """Focal loss on the max prediction within radius_px of each target peak.

    Peaks are local maxima of the Gaussian target (one per register tree). Averaged
    over trees so a missed stake is not diluted by the rest of the grid.
    """
    with _fp32_loss(pred):
        pred = pred.float().clamp(_PROB_EPS, 1.0 - _PROB_EPS)
        target = target.float()
        if ignore is not None:
            ignore = ignore.float()
        peak_k = 3
        target_pool = F.max_pool2d(target, peak_k, stride=1, padding=peak_k // 2)
        is_peak = (target >= min_target) & (target >= target_pool - 1e-6)
        if ignore is not None:
            is_peak = is_peak & (ignore < 1)
        radius_px = max(1, int(radius_px))
        window = radius_px * 2 + 1
        pred_pool = F.max_pool2d(pred, window, stride=1, padding=radius_px)
        peak_pred = pred_pool[is_peak]
        if peak_pred.numel() == 0:
            return pred.sum() * 0.0
        return (-torch.log(peak_pred) * torch.pow(1 - peak_pred, alpha)).mean()

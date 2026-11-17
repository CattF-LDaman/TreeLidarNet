import torch

from treelidarnet.models.losses import (
    focal_loss_components,
    per_tree_peak_loss,
    unknown_peak_loss,
)


def test_unknown_mask_reduces_negative_loss():
    pred = torch.full((1, 1, 4, 4), 0.5)
    target = torch.zeros(1, 1, 4, 4)
    ignore = torch.zeros(1, 1, 4, 4)
    unknown_all = torch.zeros(1, 1, 4, 4)
    unknown_none = torch.ones(1, 1, 4, 4)
    _, _, neg_labeled = focal_loss_components(pred, target, ignore, unknown=unknown_all, beta=4.0)
    _, _, neg_unknown = focal_loss_components(pred, target, ignore, unknown=unknown_none, beta=4.0)
    assert float(neg_unknown.item()) < float(neg_labeled.item())


def test_per_tree_peak_loss_penalizes_a_missed_stake():
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 8, 8] = 1.0
    target[0, 0, 24, 24] = 1.0
    pred_hit = torch.full((1, 1, 32, 32), 0.05)
    pred_hit[0, 0, 8, 8] = 0.9
    pred_hit[0, 0, 24, 24] = 0.9
    pred_miss = pred_hit.clone()
    pred_miss[0, 0, 24, 24] = 0.05
    hit = per_tree_peak_loss(pred_hit, target, radius_px=4)
    miss = per_tree_peak_loss(pred_miss, target, radius_px=4)
    assert float(miss.item()) > float(hit.item())


def test_saturated_prediction_stays_finite():
    pred = torch.ones(1, 1, 8, 8, dtype=torch.float16)
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 3, 3] = 1.0
    loss, _pos, neg = focal_loss_components(pred, target, beta=1.0)
    tree = per_tree_peak_loss(pred, target, radius_px=2)
    assert torch.isfinite(loss)
    assert torch.isfinite(neg)
    assert torch.isfinite(tree)
    assert float(tree.item()) < 1e-3


def test_per_tree_peak_loss_empty_target_is_zero():
    pred = torch.full((1, 1, 16, 16), 0.4)
    target = torch.zeros(1, 1, 16, 16)
    loss = per_tree_peak_loss(pred, target, radius_px=4)
    assert float(loss.item()) == 0.0


def test_unknown_peak_loss_lowers_mid_scores_and_spares_high_ones():
    pred = torch.full((1, 1, 32, 32), 0.05)
    pred[0, 0, 8, 8] = 0.72
    pred[0, 0, 24, 24] = 0.95
    pred.requires_grad_(True)
    unknown = torch.ones(1, 1, 32, 32)
    loss = unknown_peak_loss(
        pred, unknown, radius_px=4, keep=0.90, floor=0.20, use_focal=False
    )
    loss.backward()
    assert abs(float(loss.item()) - 0.72) < 1e-5
    assert float(pred.grad[0, 0, 8, 8].item()) > 0
    assert float(pred.grad[0, 0, 24, 24].item()) == 0.0
    assert float(pred.grad[0, 0, 0, 0].item()) == 0.0


def test_unknown_peak_loss_ignores_labeled_cells():
    pred = torch.full((1, 1, 16, 16), 0.05)
    pred[0, 0, 8, 8] = 0.70
    unknown = torch.zeros(1, 1, 16, 16)
    loss = unknown_peak_loss(pred, unknown, radius_px=2, keep=0.90, floor=0.20)
    assert float(loss.item()) == 0.0

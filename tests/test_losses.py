import torch

from treelidarnet.models.losses import focal_loss_components


def test_unknown_mask_reduces_negative_loss():
    pred = torch.full((1, 1, 4, 4), 0.5)
    target = torch.zeros(1, 1, 4, 4)
    ignore = torch.zeros(1, 1, 4, 4)
    unknown_all = torch.zeros(1, 1, 4, 4)
    unknown_none = torch.ones(1, 1, 4, 4)
    _, _, neg_labeled = focal_loss_components(pred, target, ignore, unknown=unknown_all, beta=4.0)
    _, _, neg_unknown = focal_loss_components(pred, target, ignore, unknown=unknown_none, beta=4.0)
    assert float(neg_unknown.item()) < float(neg_labeled.item())

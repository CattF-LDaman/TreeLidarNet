import torch

from treelidarnet.models.crop_metrics import crop_val_metrics


def test_crop_val_metrics_perfect_split():
    probs = torch.tensor([0.9, 0.1, 0.8, 0.2])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    m = crop_val_metrics(probs, labels, threshold=0.5)
    assert m["f1"] == 1.0
    assert m["auroc"] == 1.0


def test_collate_skips_none():
    from treelidarnet.datasets.crop_dataset import collate_crops

    assert collate_crops([None, None]) is None
    one = {
        "points": torch.zeros(512, 6),
        "label": torch.tensor(1.0),
        "crop_id": "x",
    }
    out = collate_crops([None, one])
    assert out is not None
    assert out["points"].shape[0] == 1
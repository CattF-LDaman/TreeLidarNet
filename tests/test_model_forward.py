import torch

from treelidarnet.models.gent_detector import GentTreeDetector


def test_forward_shape():
    model = GentTreeDetector(in_channels=4, base=16)
    x = torch.randn(2, 4, 200, 200)
    y = model(x)
    assert y.shape == (2, 1, 200, 200)
    assert y.min() >= 0 and y.max() <= 1

import torch

from treelidarnet.models.crop_verifier import GentCropVerifier


def test_crop_verifier_forward_backward():
    model = GentCropVerifier()
    x = torch.randn(2, 512, 6, requires_grad=True)
    logits = model(x)
    assert logits.shape == (2, 1)
    loss = logits.pow(2).mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
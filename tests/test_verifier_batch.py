"""Verifier batch scoring helpers."""

from treelidarnet.verifier_batch import _det_seed


def test_det_seed_matches_legacy():
    x, y, gi = 203_456.12, 191_234.56, 42_000
    assert _det_seed(x, y, gi) == hash((x, y, gi)) % (2**31)
    assert _det_seed(x, y, gi + 1) == hash((x, y, gi + 1)) % (2**31)
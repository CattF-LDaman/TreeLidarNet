from pathlib import Path

from treelidarnet.datasets.chip_index import (
    order_paths_outward,
    sample_weights_for_tree_oversample,
)


def test_tree_oversample_weights():
    paths = [Path("gent_50m_1_2.npz"), Path("gent_50m_3_4.npz"), Path("gent_50m_5_6.npz")]
    counts = {"gent_50m_1_2": 3, "gent_50m_3_4": 0, "gent_50m_5_6": 1}
    w = sample_weights_for_tree_oversample(paths, counts, tree_weight=10.0)
    assert w == [10.0, 1.0, 10.0]
    assert sample_weights_for_tree_oversample(paths, counts, tree_weight=1.0) == [1.0, 1.0, 1.0]


def test_order_paths_outward_starts_at_the_center():
    paths = [
        Path("gent_50m_0_0.npz"),
        Path("gent_50m_1000_0.npz"),
        Path("gent_50m_500_500.npz"),
    ]
    ordered, center = order_paths_outward(paths, chip_size_m=50.0)
    assert ordered[0].name == "gent_50m_500_500.npz"
    assert {p.name for p in ordered[1:]} == {"gent_50m_0_0.npz", "gent_50m_1000_0.npz"}
    assert center == (525.0, 275.0)

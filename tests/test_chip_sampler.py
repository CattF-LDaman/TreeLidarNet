from pathlib import Path

from treelidarnet.datasets.chip_index import sample_weights_for_tree_oversample


def test_tree_oversample_weights():
    paths = [Path("gent_50m_1_2.npz"), Path("gent_50m_3_4.npz"), Path("gent_50m_5_6.npz")]
    counts = {"gent_50m_1_2": 3, "gent_50m_3_4": 0, "gent_50m_5_6": 1}
    w = sample_weights_for_tree_oversample(paths, counts, tree_weight=10.0)
    assert w == [10.0, 1.0, 10.0]
    assert sample_weights_for_tree_oversample(paths, counts, tree_weight=1.0) == [1.0, 1.0, 1.0]

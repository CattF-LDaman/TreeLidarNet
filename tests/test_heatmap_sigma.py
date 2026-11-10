from treelidarnet.data.chips import HEATMAP_SIGMA_M, heatmap_sigma_m, rasterize_heatmap


def test_heatmap_sigma_unknown():
    assert heatmap_sigma_m(None) == HEATMAP_SIGMA_M


def test_heatmap_sigma_scales():
    small = heatmap_sigma_m(50.0)
    large = heatmap_sigma_m(300.0)
    assert small < large
    assert 1.0 <= small <= 2.5
    assert 1.0 <= large <= 2.5


def test_rasterize_per_tree_sigma():
    h = rasterize_heatmap(
        [(10.0, 10.0, 1.0), (20.0, 20.0, 2.0)],
        origin_x=0.0,
        origin_y=0.0,
    )
    assert h.max() > 0.9

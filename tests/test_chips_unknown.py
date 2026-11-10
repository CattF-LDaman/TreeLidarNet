from treelidarnet.data.chips import make_unknown_mask, rasterize_heatmap


def test_unknown_mask_near_tree_is_labeled():
    x0, y0 = 200_000.0, 193_000.0
    tx, ty = x0 + 10.0, y0 + 10.0
    unknown = make_unknown_mask([(tx, ty)], origin_x=x0, origin_y=y0, unknown_band_m=12.0)
    heat = rasterize_heatmap([(tx, ty)], origin_x=x0, origin_y=y0)
    peak = unknown[heat.argmax() // unknown.shape[1], heat.argmax() % unknown.shape[1]]
    assert peak == 0.0


def test_unknown_mask_far_from_trees():
    x0, y0 = 200_000.0, 193_000.0
    unknown = make_unknown_mask([], origin_x=x0, origin_y=y0)
    assert unknown.mean() == 1.0


def test_unknown_mask_edge_cell():
    x0, y0 = 200_000.0, 193_000.0
    tx, ty = x0 + 25.0, y0 + 25.0
    unknown = make_unknown_mask([(tx, ty)], origin_x=x0, origin_y=y0, unknown_band_m=12.0)
    assert unknown[0, 0] == 1.0

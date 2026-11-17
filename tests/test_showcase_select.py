from treelidarnet.media.showcase_select import ShowcasePick, farthest_point_select


def test_farthest_point_select_spacing():
    cands = [
        ShowcasePick(3.7, 51.05, 103500.0, 196000.0, 0.95, 0.95, "a"),
        ShowcasePick(3.71, 51.051, 103510.0, 196010.0, 0.94, 0.94, "b"),
        ShowcasePick(3.8, 51.1, 104500.0, 197000.0, 0.90, 0.90, "c"),
    ]
    picked = farthest_point_select(cands, n=2, min_spacing_m=400.0)
    assert len(picked) == 2
    assert picked[0] == cands[0]
    assert picked[1].chip_id == "c"


def test_quality_product():
    p = ShowcasePick(0, 0, 0, 0, 0.9, 0.8, "x")
    assert abs(p.quality - 0.72) < 1e-6
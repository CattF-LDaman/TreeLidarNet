import pytest

from treelidarnet.explorer.combine import (
    CombineConfig,
    detection_passes,
    export_keep_passes,
    weighted_combo,
)


@pytest.mark.parametrize(
    ("mode", "v2", "v3", "exp"),
    [
        ("either_or", 0.6, 0.3, True),
        ("either_or", 0.3, 0.6, True),
        ("either_or", 0.3, 0.3, False),
        ("both_and", 0.6, 0.6, True),
        ("both_and", 0.6, 0.3, False),
        ("disagreement", 0.8, 0.5, True),
    ],
)
def test_combine_modes(mode, v2, v3, exp):
    cfg = CombineConfig(mode=mode, bev_min=0.5, t_v2=0.5, t_v3=0.5, disagree_delta=0.15)
    assert (
        detection_passes(
            bev=0.9,
            v2=v2,
            v3=v3,
            dist_register_m=10.0,
            has_rgb=True,
            cfg=cfg,
        )
        is exp
    )


def test_export_weighted_gate():
    assert weighted_combo(0.6, 0.2) == 0.4
    assert export_keep_passes(v2=0.7, v3=0.3, combine_mode="weighted", threshold=0.45)
    assert not export_keep_passes(v2=0.5, v3=0.3, combine_mode="weighted", threshold=0.45)
    assert export_keep_passes(v2=0.5, v3=0.3, combine_mode="v3_only", threshold=0.25)
    assert not export_keep_passes(v2=0.9, v3=0.1, combine_mode="v3_only", threshold=0.5)


def test_discoveries_only():
    cfg = CombineConfig(discoveries_only=True, discovery_min_m=2.0, bev_min=0.0, t_v2=0.0, t_v3=0.0)
    assert not detection_passes(
        bev=0.9, v2=0.9, v3=0.9, dist_register_m=1.0, has_rgb=True, cfg=cfg
    )
    assert detection_passes(
        bev=0.9, v2=0.9, v3=0.9, dist_register_m=3.0, has_rgb=True, cfg=cfg
    )
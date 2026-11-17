import numpy as np

from treelidarnet.data.chips import chip_id
from treelidarnet.data.crops import (
    IN_CHANNELS,
    IN_CHANNELS_RGB,
    build_crop_tensor,
    chip_id_for_point,
    chip_origin_for_point,
    global_to_chip_local,
)


def test_chip_origin_and_local_coords():
    x0, y0 = chip_origin_for_point(103_517.3, 195_842.1)
    assert x0 == 103_500.0
    assert y0 == 195_800.0
    assert chip_id_for_point(103_517.3, 195_842.1) == chip_id(x0, y0)
    lx, ly = global_to_chip_local(103_517.3, 195_842.1, origin_x=x0, origin_y=y0)
    assert abs(lx - 17.3) < 1e-4
    assert abs(ly - 42.1) < 1e-4


def test_build_crop_tensor_centers_and_reproducible_subsample():
    n = 2000
    rng = np.random.default_rng(0)
    xyz = np.zeros((n, 3), dtype=np.float32)
    xyz[:, 0] = rng.uniform(-5, 5, size=n)
    xyz[:, 1] = rng.uniform(-5, 5, size=n)
    xyz[:, 2] = rng.uniform(0, 25, size=n)
    feat = np.stack(
        [
            rng.integers(0, 1000, size=n),
            rng.integers(1, 3, size=n),
            rng.integers(2, 5, size=n),
        ],
        axis=1,
    ).astype(np.float32)
    crop_a = build_crop_tensor(
        xyz, feat, center_x=0.0, center_y=0.0, radius_m=8.0, num_points=512, rng=np.random.default_rng(42)
    )
    crop_b = build_crop_tensor(
        xyz, feat, center_x=0.0, center_y=0.0, radius_m=8.0, num_points=512, rng=np.random.default_rng(42)
    )
    assert crop_a is not None
    assert crop_a.shape == (512, IN_CHANNELS)
    assert np.allclose(crop_a.mean(axis=0)[:2], 0.0, atol=0.5)
    assert np.allclose(crop_a, crop_b)


def test_build_crop_tensor_with_rgb():
    n = 500
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-5, 5, size=(n, 3)).astype(np.float32)
    xyz[:, 2] = rng.uniform(0, 20, size=n)
    feat = rng.integers(0, 1000, size=(n, 3)).astype(np.float32)
    rgb = rng.integers(0, 65535, size=(n, 3)).astype(np.float32)
    crop = build_crop_tensor(
        xyz,
        feat,
        center_x=0.0,
        center_y=0.0,
        radius_m=8.0,
        num_points=64,
        rgb=rgb,
        in_channels=IN_CHANNELS_RGB,
        rng=np.random.default_rng(0),
    )
    assert crop is not None
    assert crop.shape == (64, IN_CHANNELS_RGB)
    assert crop[:, 6:9].max() <= 1.0


def test_build_crop_tensor_sparse_returns_none():
    xyz = np.array([[1.0, 1.0, 2.0]], dtype=np.float32)
    feat = np.array([[100.0, 1.0, 2.0]], dtype=np.float32)
    assert build_crop_tensor(xyz, feat, center_x=0.0, center_y=0.0, min_points=32) is None
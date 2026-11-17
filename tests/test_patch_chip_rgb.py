"""RGB chip patch helpers (matching + DTM sampling)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, rowcol

from treelidarnet.data.ground import sample_dtm_at_points
from treelidarnet.data.patch_chip_rgb import _chip_id_list, _match_rgb_to_chip_xyz


def _sample_dtm_loop(dtm_path: Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ground = np.full(len(x), np.nan, dtype=np.float64)
    if not dtm_path.is_file():
        return ground
    with rasterio.open(dtm_path) as src:
        band = src.read(1)
        for i in range(len(x)):
            try:
                r, c = rowcol(src.transform, x[i], y[i])
            except Exception:
                continue
            if 0 <= r < src.height and 0 <= c < src.width:
                val = band[r, c]
                if val != src.nodata:
                    ground[i] = float(val)
    return ground


def test_sample_dtm_matches_reference_loop(tmp_path: Path) -> None:
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    transform = Affine.translation(1000.0, 2000.0) * Affine.scale(1.0, -1.0)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": 10,
        "height": 10,
        "count": 1,
        "transform": transform,
        "nodata": -9999.0,
    }
    path = tmp_path / "dtm.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    x = np.array([1000.5, 1005.5, 1012.0, 999.0, 1010.5])
    y = np.array([2000.5, 1995.5, 1990.0, 2001.0, 1992.5])
    ref = _sample_dtm_loop(path, x, y)
    got = sample_dtm_at_points(path, x, y)
    np.testing.assert_allclose(got, ref, equal_nan=True)


def test_match_rgb_last_wins_and_threshold() -> None:
    cloud = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    rgb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    chip = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    out = _match_rgb_to_chip_xyz(chip, cloud, rgb, tol=0.01)
    assert out is not None
    np.testing.assert_allclose(out[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(out[1], [0.0, 0.0, 1.0])

    extra = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    cloud_wide = np.concatenate([cloud, extra], axis=0)
    rgb_wide = np.concatenate([rgb, [[0.2, 0.2, 0.2]]], axis=0)
    chip_sparse = np.array(
        [[99.0, 99.0, 99.0], [98.0, 98.0, 98.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    assert _match_rgb_to_chip_xyz(chip_sparse, cloud_wide, rgb_wide, tol=0.01) is None


def test_match_rgb_xy_fallback_when_h_agl_drifts() -> None:
    cloud = np.array(
        [[1.0, 2.0, 5.0], [1.0, 2.0, 5.0], [3.0, 4.0, 1.0]],
        dtype=np.float32,
    )
    rgb = np.array(
        [[10.0, 20.0, 30.0], [0.0, 1.0, 0.0], [1.0, 2.0, 3.0]],
        dtype=np.float32,
    )
    chip = np.array(
        [[1.0, 2.0, 5.05], [3.0, 4.0, 0.92]],
        dtype=np.float32,
    )
    out = _match_rgb_to_chip_xyz(chip, cloud, rgb, tol=0.01)
    assert out is not None
    np.testing.assert_allclose(out[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(out[1], [1.0, 2.0, 3.0])


def test_match_rgb_dict_parity_on_random() -> None:
    rng = np.random.default_rng(0)
    cloud = rng.uniform(-5, 5, (200, 3)).astype(np.float32)
    chip = cloud[rng.choice(200, size=180, replace=False)]
    rgb = rng.uniform(0, 1, (200, 3)).astype(np.float32)

    keys_cloud = np.round(cloud / 0.01).astype(np.int64)
    keys_chip = np.round(chip / 0.01).astype(np.int64)
    lut: dict[tuple[int, int, int], int] = {}
    for i, key in enumerate(keys_cloud):
        lut[tuple(key.tolist())] = i
    ref = np.zeros((len(chip), 3), dtype=np.float32)
    for i, key in enumerate(keys_chip):
        j = lut.get(tuple(key.tolist()))
        if j is None:
            continue
        ref[i] = rgb[j]

    out = _match_rgb_to_chip_xyz(chip, cloud, rgb, tol=0.01)
    assert out is not None
    np.testing.assert_allclose(out, ref)


def test_chip_id_list_register_only(tmp_path: Path) -> None:
    import pandas as pd

    path = tmp_path / "crops_index.parquet"
    pd.DataFrame(
        {
            "chip_id": ["chip_a", "chip_a", "chip_b", "chip_c"],
            "source": ["register", "register", "grb", "register"],
        }
    ).to_parquet(path)
    all_ids = _chip_id_list(
        crops_index=path, chips_index=None, all_chips=False, register_chips=False
    )
    assert all_ids == ["chip_a", "chip_b", "chip_c"]
    reg_ids = _chip_id_list(
        crops_index=path, chips_index=None, all_chips=False, register_chips=True
    )
    assert reg_ids == ["chip_a", "chip_c"]
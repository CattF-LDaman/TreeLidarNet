"""Pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from treelidarnet.data.chips import grid_cells, make_ignore_mask, rasterize_heatmap

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def synthetic_chip_npz(tmp_path_factory) -> Path:
    n = grid_cells()
    x0, y0 = 200_000.0, 193_000.0
    xyz = np.array([[10.0, 10.0, 5.0], [20.0, 15.0, 8.0]], dtype=np.float32)
    feat = np.array([[100, 1, 1], [120, 2, 1]], dtype=np.float32)
    heat = rasterize_heatmap([(x0 + 10, y0 + 10)], origin_x=x0, origin_y=y0)
    ignore = make_ignore_mask(n, n)
    path = FIXTURES / "synthetic_chip.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xyz=xyz,
        feat=feat,
        heatmap=heat,
        ignore=ignore,
        origin=np.array([x0, y0], dtype=np.float64),
    )
    return path

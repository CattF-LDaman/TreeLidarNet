from pathlib import Path

import numpy as np
import pandas as pd

from treelidarnet.data.chips import grid_cells, make_ignore_mask, rasterize_heatmap
from treelidarnet.data.patch_heatmaps import patch_chips_clear_ignore
from treelidarnet.datasets.chip_dataset import ChipDataset

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_chip.npz"


def test_ignore_mask_trains_every_cell():
    mask = make_ignore_mask(200, 200)
    assert mask.shape == (200, 200)
    assert mask.dtype == np.uint8
    assert int(mask.sum()) == 0


def test_patch_clears_stored_ignore(tmp_path, monkeypatch):
    monkeypatch.setattr("treelidarnet.datasets.chip_index.chips_dir", lambda: tmp_path)
    ring = np.zeros((4, 4), dtype=np.uint8)
    ring[1, :] = 1
    clear = np.zeros((4, 4), dtype=np.uint8)
    np.savez_compressed(tmp_path / "gent_50m_0_0.npz", ignore=ring, heatmap=clear)
    np.savez_compressed(tmp_path / "gent_50m_50_0.npz", ignore=clear, heatmap=clear)
    index = tmp_path / "index.parquet"
    pd.DataFrame({"chip_id": ["gent_50m_0_0", "gent_50m_50_0"]}).to_parquet(index)
    assert patch_chips_clear_ignore(index_parquet=index) == 1
    with np.load(tmp_path / "gent_50m_0_0.npz") as data:
        assert int(data["ignore"].sum()) == 0
        assert data["heatmap"].shape == (4, 4)


def test_chip_dataset_item():
    n = grid_cells()
    x0, y0 = 200_000.0, 193_000.0
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIXTURE,
        xyz=np.array([[5.0, 5.0, 3.0]], dtype=np.float32),
        feat=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        heatmap=rasterize_heatmap([(x0 + 5, y0 + 5)], origin_x=x0, origin_y=y0),
        ignore=make_ignore_mask(n, n),
        origin=np.array([x0, y0]),
    )
    ds = ChipDataset([FIXTURE])
    batch = ds[0]
    assert batch["bev"].shape[0] == 4
    assert batch["heatmap"].shape == (1, n, n)

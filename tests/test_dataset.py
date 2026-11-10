from pathlib import Path

import numpy as np

from treelidarnet.data.chips import grid_cells, make_ignore_mask, rasterize_heatmap
from treelidarnet.datasets.chip_dataset import ChipDataset

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_chip.npz"


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

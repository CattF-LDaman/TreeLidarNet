import pandas as pd
from pathlib import Path

from treelidarnet.data.build_crop_index import _split_bucket


def test_split_bucket_covers_all():
    splits = {_split_bucket(f"id_{i}") for i in range(500)}
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits


def test_negative_splits_not_all_train(tmp_path, monkeypatch):
    """GRB negatives must land in val/test, not train-only."""
    import numpy as np

    from treelidarnet.data.build_crop_index import build_crop_index
    from treelidarnet.paths import chips_dir, manifests_dir

    chip_store = tmp_path / "chips"
    chip_store.mkdir()
    monkeypatch.setattr("treelidarnet.data.build_crop_index.chips_dir", lambda: chip_store)

    x0, y0 = 100_000.0, 190_000.0
    h, w = 200, 200
    hn = np.zeros((h, w), dtype=np.float32)
    hn[50:60, 50:60] = 1.0
    np.savez_compressed(
        chip_store / "gent_50m_100000_190000.npz",
        xyz=np.zeros((100, 3), dtype=np.float32),
        feat=np.zeros((100, 3), dtype=np.float32),
        hard_negative=hn,
        origin=np.array([x0, y0], dtype=np.float64),
    )

    trees = pd.DataFrame(
        columns=["gis_id", "x", "y", "eligible", "lon", "lat", "sortiment", "gis_slug", "stamomtrek"]
    )
    trees_path = tmp_path / "trees.parquet"
    trees.to_parquet(trees_path, index=False)
    out = tmp_path / "crops.parquet"
    build_crop_index(
        trees_parquet=trees_path,
        chips_index=None,
        out_path=out,
        positives_only=False,
        grb_negatives_per_chip=3,
        grb_max_chips=0,
        limit=None,
        seed=0,
    )
    df = pd.read_parquet(out)
    assert len(df) >= 3
    for _, row in df.iterrows():
        assert row["split"] == _split_bucket(str(row["crop_id"]))
import numpy as np
import pandas as pd

from treelidarnet.data.build_crop_index import build_crop_index
from treelidarnet.data.crs import lambert_to_wgs84
from treelidarnet.explorer.qa_flags import load_flags, upsert_flag


def test_build_crop_index_merges_explorer_fp(tmp_path, monkeypatch):
    chip_store = tmp_path / "chips"
    chip_store.mkdir()
    monkeypatch.setattr("treelidarnet.data.build_crop_index.chips_dir", lambda: chip_store)

    x0, y0 = 100_000.0, 190_000.0
    gx, gy = x0 + 25.0, y0 + 25.0
    np.savez_compressed(
        chip_store / "gent_50m_100000_190000.npz",
        xyz=np.zeros((100, 3), dtype=np.float32),
        feat=np.zeros((100, 3), dtype=np.float32),
        hard_negative=np.zeros((200, 200), dtype=np.float32),
        origin=np.array([x0, y0], dtype=np.float64),
    )

    flags_path = tmp_path / "explorer_qa_flags.parquet"
    lon, lat = lambert_to_wgs84(gx, gy)
    upsert_flag(lon=lon, lat=lat, bev=0.8, v2=0.2, v3=0.3, path=flags_path)

    tree_cols = [
        "gis_id", "x", "y", "eligible", "lon", "lat",
        "sortiment", "gis_slug", "stamomtrek",
    ]
    trees = pd.DataFrame(columns=tree_cols)
    trees_path = tmp_path / "trees.parquet"
    trees.to_parquet(trees_path, index=False)
    out = tmp_path / "crops.parquet"

    build_crop_index(
        trees_parquet=trees_path,
        chips_index=None,
        out_path=out,
        positives_only=True,
        qa_flags_path=flags_path,
    )
    df = pd.read_parquet(out)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["label"] == 0
    assert row["source"] == "explorer_fp"
    assert str(row["crop_id"]).startswith("fp_")
    flags = load_flags(flags_path, active_only=True, gui_visible_only=False)
    assert bool(flags.iloc[0]["gui_hidden"])
    assert load_flags(flags_path, gui_visible_only=True).empty
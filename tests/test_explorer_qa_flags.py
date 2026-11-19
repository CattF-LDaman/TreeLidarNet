from pathlib import Path

import pandas as pd

from treelidarnet.data.crs import lambert_to_wgs84, wgs84_to_lambert
from treelidarnet.explorer.qa_flags import (
    clear_flag,
    flag_id_for_lonlat,
    load_flags,
    upsert_flag,
)


def test_flag_id_stable_for_rounded_lonlat():
    lon, lat = 3.72123456789, 51.05432109876
    a = flag_id_for_lonlat(lon, lat)
    b = flag_id_for_lonlat(lon + 1e-8, lat - 1e-8)
    assert a == b
    assert a.startswith("fp_")


def test_upsert_dedups_by_lonlat(tmp_path: Path):
    path = tmp_path / "flags.parquet"
    lon, lat = 3.721, 51.054
    r1 = upsert_flag(lon=lon, lat=lat, bev=0.9, path=path)
    r2 = upsert_flag(lon=lon, lat=lat, bev=0.1, path=path)
    assert r1["flag_id"] == r2["flag_id"]
    active = load_flags(path, active_only=True)
    assert len(active) == 1
    assert float(active.iloc[0]["bev"]) == 0.1


def test_clear_flag_soft_deletes(tmp_path: Path):
    path = tmp_path / "flags.parquet"
    row = upsert_flag(lon=3.72, lat=51.05, path=path)
    assert clear_flag(row["flag_id"], path=path)
    assert load_flags(path, active_only=True).empty
    all_rows = load_flags(path, active_only=False)
    assert len(all_rows) == 1
    assert not bool(all_rows.iloc[0]["active"])


def test_upsert_stores_lambert_xy(tmp_path: Path):
    path = tmp_path / "flags.parquet"
    x, y = 100_025.0, 190_025.0
    lon, lat = lambert_to_wgs84(x, y)
    row = upsert_flag(lon=lon, lat=lat, path=path)
    ex, ey = wgs84_to_lambert(lon, lat)
    assert abs(row["x"] - ex) < 0.05
    assert abs(row["y"] - ey) < 0.05
    df = pd.read_parquet(path)
    assert abs(float(df.iloc[0]["x"]) - x) < 1.0
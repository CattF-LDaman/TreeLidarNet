from pathlib import Path

from treelidarnet.data.build_gent import index_laz_files, laz_paths_for_chip
from treelidarnet.data.build_gent_grid import _loaded_cloud
from treelidarnet.data.download_laz import strip_probe_points


def test_index_ignores_copc_sidecars(tmp_path: Path):
    corner = (103500, 195500)
    good = "LiDAR_DHMV_2_P3_FU_22011_2_103500_195500.laz"
    copc = "LiDAR_DHMV_2_P3_FU_22011_2_103500_195500.copc.laz"
    (tmp_path / good).write_bytes(b"x")
    (tmp_path / copc).write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 32)
    index = index_laz_files(tmp_path)
    assert [p.name for p in index[corner]] == [good]


def test_index_keeps_every_flight_line(tmp_path: Path):
    corner = (103500, 195500)
    names = [
        "LiDAR_DHMV_2_P3_FU_22011_2_103500_195500.laz",
        "LiDAR_DHMV_2_P3_FU_22010_2_103500_195500.laz",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    index = index_laz_files(tmp_path)
    assert [p.name for p in index[corner]] == sorted(names)
    paths = laz_paths_for_chip(103700, 195950, index)
    assert {p.name for p in paths} == set(names)


def test_unreadable_laz_is_skipped(tmp_path: Path):
    bad = tmp_path / "LiDAR_DHMV_2_P3_FU_22011_2_103500_195500.copc.laz"
    bad.write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 32)
    assert _loaded_cloud(bad) is None


def test_strip_probes_include_the_northern_edge():
    points = strip_probe_points(103500, 195500)
    assert (103750.0, 195750.0) in points
    assert any(y >= 195980 for _x, y in points)

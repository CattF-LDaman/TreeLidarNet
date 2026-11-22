from treelidarnet.data.chips import CHIP_SIZE_M, chip_id, grid_cells
from treelidarnet.data.crops import chip_id_for_point
from treelidarnet.viz.pipeline_chip_gif import sweep_thresholds
from treelidarnet.viz.pipeline_chip_gif import (
    ScoredPeak,
    chip_id_at_lambert,
    chip_px_to_axes_uv,
    detections_to_chip_px,
    global_to_chip_px,
)


def test_chip_id_at_lambert_matches_floor_origin():
    x0, y0 = 103850.0, 194200.0
    x, y = x0 + 12.0, y0 + 37.0
    assert chip_id_at_lambert(x, y) == chip_id_for_point(x, y)
    assert chip_id_at_lambert(x, y) == chip_id(x0, y0)


def test_global_to_chip_px_roundtrip_grid_cell():
    origin_x, origin_y = 100_000.0, 200_000.0
    gs = grid_cells()
    px, py = 40.5, 120.25
    x = origin_x + (px + 0.5) * 0.25
    y = origin_y + (py + 0.5) * 0.25
    back_px, back_py = global_to_chip_px(x, y, origin_x=origin_x, origin_y=origin_y, grid_size=gs)
    assert abs(back_px - px) < 1e-9
    assert abs(back_py - py) < 1e-9


def test_chip_px_to_axes_uv_flips_y():
    u, v = chip_px_to_axes_uv(100.0, 0.0, 200)
    assert u == 0.5
    assert v == 1.0


def test_detections_to_chip_px_shape():
    peaks = [ScoredPeak(1.0, 2.0, 0.9), ScoredPeak(3.0, 4.0, 0.8)]
    arr = detections_to_chip_px(peaks, origin_x=0.0, origin_y=0.0, grid_size=200)
    assert arr.shape == (2, 2)


def test_sweep_combo_act_frame_count():
    th = sweep_thresholds(0.45, 0.7, 0.02)
    assert abs(th[0] - 0.45) < 1e-9
    assert th[-1] >= 0.69
    assert len(th) >= 10
    assert th == sorted(th)


def test_chip_floor_snap_50m():
    cid = chip_id_at_lambert(CHIP_SIZE_M * 2.1, CHIP_SIZE_M * 3.9)
    assert cid == chip_id(100.0, 150.0)
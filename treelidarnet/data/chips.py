"""Chip IDs and heatmap targets."""

from __future__ import annotations

import numpy as np

CHIP_SIZE_M = 50.0
CELL_SIZE_M = 0.25
EDGE_IGNORE_INNER_M = 8.0
EDGE_IGNORE_OUTER_M = 15.0
HEATMAP_SIGMA_M = 1.0


def chip_id(x0: float, y0: float) -> str:
    return f"gent_50m_{int(round(x0))}_{int(round(y0))}"


def grid_cells(chip_size_m: float = CHIP_SIZE_M, cell_size_m: float = CELL_SIZE_M) -> int:
    return int(round(chip_size_m / cell_size_m))


def make_ignore_mask(
    h: int,
    w: int,
    *,
    cell_size_m: float = CELL_SIZE_M,
    inner_m: float = EDGE_IGNORE_INNER_M,
    outer_m: float = EDGE_IGNORE_OUTER_M,
) -> np.ndarray:
    """1 = ignore loss, 0 = valid."""
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    px = (xx - cx) * cell_size_m
    py = (yy - cy) * cell_size_m
    half = chip_size_from_hw(h, w, cell_size_m) / 2.0
    edge_dist = half - np.maximum(np.abs(px), np.abs(py))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[edge_dist < inner_m] = 0
    annulus = (edge_dist >= inner_m) & (edge_dist < outer_m)
    mask[annulus] = 1
    return mask


def chip_size_from_hw(h: int, w: int, cell_size_m: float) -> float:
    return max(h, w) * cell_size_m


def rasterize_heatmap(
    tree_xy: list[tuple[float, float]],
    *,
    origin_x: float,
    origin_y: float,
    chip_size_m: float = CHIP_SIZE_M,
    cell_size_m: float = CELL_SIZE_M,
    sigma_m: float = HEATMAP_SIGMA_M,
) -> np.ndarray:
    n = grid_cells(chip_size_m, cell_size_m)
    heat = np.zeros((n, n), dtype=np.float32)
    if not tree_xy:
        return heat
    yy, xx = np.mgrid[0:n, 0:n]
    cell_x = origin_x + (xx + 0.5) * cell_size_m
    cell_y = origin_y + (yy + 0.5) * cell_size_m
    for tx, ty in tree_xy:
        d2 = (cell_x - tx) ** 2 + (cell_y - ty) ** 2
        heat = np.maximum(heat, np.exp(-d2 / (2 * sigma_m * sigma_m)).astype(np.float32))
    return heat

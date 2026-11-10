"""GRB spatial index and hard-negative rasterization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import make_valid
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree

from treelidarnet.data.chips import CELL_SIZE_M, CHIP_SIZE_M, grid_cells
from treelidarnet.data.crs import _WGS84_TO_LAMBERT


@dataclass
class GrbIndex:
    geoms: list
    tree: STRtree

    @classmethod
    def from_dir(cls, grb_dir: Path, *, road_buffer_m: float = 2.0) -> GrbIndex:
        buildings = grb_dir / "gent_grb_buildings.geojson"
        roads = grb_dir / "gent_grb_roads.geojson"
        if not buildings.is_file() or not roads.is_file():
            raise FileNotFoundError(
                f"Need {buildings.name} and {roads.name} under {grb_dir} "
                "(run data-download-grb)"
            )
        geoms: list = []
        for path, kind in ((buildings, "building"), (roads, "road")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for feat in data.get("features") or []:
                g = shape(feat.get("geometry") or {})
                if g.is_empty:
                    continue
                cx, cy = g.centroid.x, g.centroid.y
                if -15.0 < cx < 15.0 and 49.0 < cy < 53.0:
                    g = shp_transform(_WGS84_TO_LAMBERT.transform, g)
                g = make_valid(g)
                if kind == "road" and road_buffer_m > 0:
                    g = g.buffer(road_buffer_m, cap_style=2, join_style=2)
                if not g.is_empty:
                    geoms.append(g)
        tree = STRtree(geoms)
        return cls(geoms=geoms, tree=tree)

    def geoms_in_chip(self, x0: float, y0: float) -> list:
        chip = box(x0, y0, x0 + CHIP_SIZE_M, y0 + CHIP_SIZE_M)
        idxs = self.tree.query(chip, predicate="intersects")
        if isinstance(idxs, np.ndarray):
            idxs = idxs.tolist()
        return [self.geoms[i] for i in idxs]


def make_hard_negative_mask(
    grb: GrbIndex | None,
    *,
    origin_x: float,
    origin_y: float,
    heatmap: np.ndarray | None = None,
    chip_size_m: float = CHIP_SIZE_M,
    cell_size_m: float = CELL_SIZE_M,
) -> np.ndarray:
    """1 = GRB road/building (strong negative), 0 elsewhere."""
    n = grid_cells(chip_size_m, cell_size_m)
    if grb is None:
        return np.zeros((n, n), dtype=np.float32)
    candidates = grb.geoms_in_chip(origin_x, origin_y)
    if not candidates:
        mask = np.zeros((n, n), dtype=np.float32)
    else:
        north = origin_y + chip_size_m
        transform = from_origin(origin_x, north, cell_size_m, cell_size_m)
        shapes = [(g, 1) for g in candidates]
        raw = rasterize(
            shapes,
            out_shape=(n, n),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        mask = np.flipud(raw).astype(np.float32)
    if heatmap is not None:
        mask = mask * (heatmap < 0.99).astype(np.float32)
    return mask

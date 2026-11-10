"""Build training chips from LAZ + gent_trees.parquet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from treelidarnet.data.chips import (
    CHIP_SIZE_M,
    chip_id,
    grid_cells,
    heatmap_sigma_m,
    make_ignore_mask,
    make_unknown_mask,
    rasterize_heatmap,
)
from treelidarnet.data.grb import GrbIndex, make_hard_negative_mask
from treelidarnet.data.ground import height_above_ground
from treelidarnet.data.laz_io import read_laz_bbox
from treelidarnet.paths import chips_dir, manifests_dir


def _trees_in_chip(df: pd.DataFrame, x0: float, y0: float, size: float) -> pd.DataFrame:
    x1, y1 = x0 + size, y0 + size
    m = (
        df["eligible"]
        & (df["x"] >= x0)
        & (df["x"] < x1)
        & (df["y"] >= y0)
        & (df["y"] < y1)
    )
    return df.loc[m]


def build_chip(
    *,
    x0: float,
    y0: float,
    laz_paths: list[Path],
    trees_df: pd.DataFrame,
    dtm_path: Path | None,
    out_dir: Path | None = None,
    grb_index: GrbIndex | None = None,
) -> tuple[Path, dict]:
    out_dir = out_dir or chips_dir()
    x1, y1 = x0 + CHIP_SIZE_M, y0 + CHIP_SIZE_M
    xyz, feat = read_laz_bbox(laz_paths, xmin=x0, ymin=y0, xmax=x1, ymax=y1)
    h_agl, qc = height_above_ground(xyz, feat, dtm_path=dtm_path)
    n = grid_cells()
    chip_trees = _trees_in_chip(trees_df, x0, y0, CHIP_SIZE_M)
    tree_xy = [
        (float(r.x), float(r.y), heatmap_sigma_m(r.get("stamomtrek")))
        for r in chip_trees.to_dict("records")
    ]
    heatmap = rasterize_heatmap(tree_xy, origin_x=x0, origin_y=y0)
    ignore = make_ignore_mask(n, n)
    unknown = make_unknown_mask(tree_xy, origin_x=x0, origin_y=y0)
    hard_negative = make_hard_negative_mask(
        grb_index, origin_x=x0, origin_y=y0, heatmap=heatmap
    )
    if len(xyz):
        local = xyz.copy()
        local[:, 0] -= x0
        local[:, 1] -= y0
        local[:, 2] = h_agl
    else:
        local = np.zeros((0, 3), dtype=np.float32)

    cid = chip_id(x0, y0)
    npz_path = out_dir / f"{cid}.npz"
    np.savez_compressed(
        npz_path,
        xyz=local.astype(np.float32),
        feat=feat.astype(np.float32),
        heatmap=heatmap,
        ignore=ignore,
        unknown=unknown,
        hard_negative=hard_negative,
        origin=np.array([x0, y0], dtype=np.float64),
    )
    pts_per_m2 = len(xyz) / (CHIP_SIZE_M * CHIP_SIZE_M) if len(xyz) else 0.0
    meta = {
        "chip_id": cid,
        "origin_x": x0,
        "origin_y": y0,
        "point_count": int(len(xyz)),
        "mean_pts_per_m2": pts_per_m2,
        "tree_count": int(len(chip_trees)),
        "qc": qc,
    }
    sidecar = out_dir / f"{cid}.meta.json"
    sidecar.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return npz_path, meta


def build_chips_for_origins(
    origins: list[tuple[float, float]],
    *,
    laz_paths: list[Path],
    trees_parquet: Path,
    dtm_path: Path | None,
    index_out: Path | None = None,
    grb_index: GrbIndex | None = None,
) -> pd.DataFrame:
    trees_df = pd.read_parquet(trees_parquet)
    rows = []
    for x0, y0 in origins:
        _, meta = build_chip(
            x0=x0,
            y0=y0,
            laz_paths=laz_paths,
            trees_df=trees_df,
            dtm_path=dtm_path,
            grb_index=grb_index,
        )
        rows.append(meta)
    idx = pd.DataFrame(rows)
    index_out = index_out or (manifests_dir() / "chips_index.parquet")
    idx.to_parquet(index_out, index=False)
    return idx


def discover_laz_dir(laz_dir: Path) -> list[Path]:
    return sorted(laz_dir.glob("*.laz")) + sorted(laz_dir.glob("*.LAZ"))

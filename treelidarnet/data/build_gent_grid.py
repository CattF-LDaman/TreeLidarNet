"""Build full 50 m grid chips over Gent bbox (canopy v2 coverage)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from treelidarnet.data.build_chips import build_chip
from treelidarnet.data.build_gent import index_laz_files, laz_paths_for_chip
from treelidarnet.data.chips import CHIP_SIZE_M, chip_id
from treelidarnet.data.grb import GrbIndex
from treelidarnet.paths import chips_dir, data_dir, manifests_dir


def grid_origins_from_parquet(
    trees_parquet: Path,
    *,
    stride_m: float = CHIP_SIZE_M,
) -> list[tuple[float, float]]:
    df = pd.read_parquet(trees_parquet)
    xmin = float(np.floor(df["x"].min() / stride_m) * stride_m)
    xmax = float(np.ceil(df["x"].max() / stride_m) * stride_m)
    ymin = float(np.floor(df["y"].min() / stride_m) * stride_m)
    ymax = float(np.ceil(df["y"].max() / stride_m) * stride_m)
    origins: list[tuple[float, float]] = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            origins.append((x, y))
            y += stride_m
        x += stride_m
    return origins


def build_gent_grid_chips(
    *,
    trees_parquet: Path,
    laz_dir: Path,
    dtm_path: Path | None = None,
    limit: int | None = None,
    stride_m: float = CHIP_SIZE_M,
    grb_dir: Path | None = None,
    road_buffer_m: float = 2.0,
    skip_existing: bool = False,
) -> pd.DataFrame:
    index = index_laz_files(laz_dir)
    if not index:
        raise FileNotFoundError(f"No indexed LAZ in {laz_dir}")
    grb_index: GrbIndex | None = None
    if grb_dir is None:
        default_grb = data_dir() / "raw" / "grb"
        if (default_grb / "gent_grb_buildings.geojson").is_file():
            grb_dir = default_grb
    if grb_dir is not None:
        grb_index = GrbIndex.from_dir(grb_dir, road_buffer_m=road_buffer_m)
        print(f"GRB hard negatives: {grb_dir}", flush=True)
    origins = grid_origins_from_parquet(trees_parquet, stride_m=stride_m)
    if limit:
        origins = origins[:limit]
    trees_df = pd.read_parquet(trees_parquet)
    rows = []
    skipped = 0
    reused = 0
    for i, (x0, y0) in enumerate(origins, 1):
        if skip_existing:
            cid = chip_id(x0, y0)
            npz_path = chips_dir() / f"{cid}.npz"
            if npz_path.is_file():
                meta_path = chips_dir() / f"{cid}.meta.json"
                if meta_path.is_file():
                    rows.append(json.loads(meta_path.read_text(encoding="utf-8")))
                reused += 1
                if i % 200 == 0:
                    print(
                        f"grid chips {i}/{len(origins)} built={len(rows)} "
                        f"skip={skipped} reused={reused}",
                        flush=True,
                    )
                continue
        laz_paths = laz_paths_for_chip(x0, y0, index)
        if not laz_paths:
            skipped += 1
            continue
        _, meta = build_chip(
            x0=x0,
            y0=y0,
            laz_paths=laz_paths,
            trees_df=trees_df,
            dtm_path=dtm_path,
            grb_index=grb_index,
        )
        rows.append(meta)
        if i % 200 == 0:
            print(
                f"grid chips {i}/{len(origins)} built={len(rows)} skip={skipped} reused={reused}",
                flush=True,
            )
    idx = pd.DataFrame(rows)
    out = manifests_dir() / "chips_index_full.parquet"
    idx.to_parquet(out, index=False)
    print(
        f"Done: {len(rows)} index rows, skipped {skipped} (no LAZ), reused {reused} existing npz",
        flush=True,
    )
    print(f"Index: {out}", flush=True)
    return idx

"""Build full 50 m grid chips over Gent bbox (canopy v2 coverage)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from treelidarnet.data.build_chips import build_chip
from treelidarnet.data.build_gent import index_laz_files, laz_paths_for_chip
from treelidarnet.data.chips import CHIP_SIZE_M
from treelidarnet.paths import manifests_dir


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
) -> pd.DataFrame:
    index = index_laz_files(laz_dir)
    if not index:
        raise FileNotFoundError(f"No indexed LAZ in {laz_dir}")
    origins = grid_origins_from_parquet(trees_parquet, stride_m=stride_m)
    if limit:
        origins = origins[:limit]
    trees_df = pd.read_parquet(trees_parquet)
    rows = []
    skipped = 0
    for i, (x0, y0) in enumerate(origins, 1):
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
        )
        rows.append(meta)
        if i % 200 == 0:
            print(f"grid chips {i}/{len(origins)} built={len(rows)} skip={skipped}", flush=True)
    idx = pd.DataFrame(rows)
    out = manifests_dir() / "chips_index_full.parquet"
    idx.to_parquet(out, index=False)
    print(f"Done: {len(rows)} grid chips, skipped {skipped} (no LAZ)", flush=True)
    print(f"Index: {out}", flush=True)
    return idx

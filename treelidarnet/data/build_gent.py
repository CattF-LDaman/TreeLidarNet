"""Build all tree-centered chips for Gent."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from treelidarnet.data.build_chips import build_chip
from treelidarnet.data.chips import CHIP_SIZE_M
from treelidarnet.data.grb import GrbIndex
from treelidarnet.paths import data_dir, manifests_dir


def index_laz_files(laz_dir: Path) -> dict[tuple[int, int], Path]:
    index: dict[tuple[int, int], Path] = {}
    for p in laz_dir.glob("*.laz"):
        m = re.search(r"_(\d+)_(\d+)\.laz$", p.name, re.I)
        if m:
            index[(int(m.group(1)), int(m.group(2)))] = p
    return index


def laz_paths_for_chip(x0: float, y0: float, index: dict[tuple[int, int], Path]) -> list[Path]:
    x1, y1 = x0 + CHIP_SIZE_M, y0 + CHIP_SIZE_M
    keys = set()
    for tx in {int(np.floor(x0 / 500) * 500), int(np.floor((x1 - 1e-6) / 500) * 500)}:
        for ty in {int(np.floor(y0 / 500) * 500), int(np.floor((y1 - 1e-6) / 500) * 500)}:
            keys.add((tx, ty))
    return [index[k] for k in keys if k in index]


def unique_chip_origins(trees_parquet: Path) -> list[tuple[float, float]]:
    df = pd.read_parquet(trees_parquet)
    df = df[df["eligible"]]
    x0 = np.floor(df.x.to_numpy() / CHIP_SIZE_M) * CHIP_SIZE_M
    y0 = np.floor(df.y.to_numpy() / CHIP_SIZE_M) * CHIP_SIZE_M
    return sorted({(float(a), float(b)) for a, b in zip(x0, y0, strict=False)})


def build_gent_chips(
    *,
    trees_parquet: Path,
    laz_dir: Path,
    dtm_path: Path | None = None,
    limit: int | None = None,
    grb_dir: Path | None = None,
    road_buffer_m: float = 2.0,
) -> pd.DataFrame:
    index = index_laz_files(laz_dir)
    if not index:
        raise FileNotFoundError(f"No indexed LAZ in {laz_dir}")
    origins = unique_chip_origins(trees_parquet)
    if limit:
        origins = origins[:limit]
    trees_df = pd.read_parquet(trees_parquet)
    grb_index: GrbIndex | None = None
    if grb_dir is None:
        default_grb = data_dir() / "raw" / "grb"
        if (default_grb / "gent_grb_buildings.geojson").is_file():
            grb_dir = default_grb
    if grb_dir is not None:
        grb_index = GrbIndex.from_dir(grb_dir, road_buffer_m=road_buffer_m)
        print(f"GRB hard negatives: {grb_dir}", flush=True)
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
            grb_index=grb_index,
        )
        rows.append(meta)
        if i % 200 == 0:
            print(f"chips {i}/{len(origins)} built={len(rows)} skip={skipped}", flush=True)
    idx = pd.DataFrame(rows)
    out = manifests_dir() / "chips_index.parquet"
    idx.to_parquet(out, index=False)
    print(f"Done: {len(rows)} chips, skipped {skipped} (no LAZ)", flush=True)
    return idx

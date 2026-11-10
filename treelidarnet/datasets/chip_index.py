"""Resolve chip npz paths from manifest parquets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from treelidarnet.paths import chips_dir


def chip_paths_from_index(index_parquet: Path) -> list[Path]:
    df = pd.read_parquet(index_parquet)
    col = "chip_id" if "chip_id" in df.columns else df.columns[0]
    paths: list[Path] = []
    for cid in df[col].astype(str):
        p = chips_dir() / f"{cid}.npz"
        if p.is_file():
            paths.append(p)
    return sorted(paths)

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


def tree_counts_by_chip_id(index_parquet: Path) -> dict[str, int]:
    df = pd.read_parquet(index_parquet)
    col = "chip_id" if "chip_id" in df.columns else df.columns[0]
    if "tree_count" not in df.columns:
        return {str(cid): 0 for cid in df[col].astype(str)}
    return {
        str(cid): int(tc)
        for cid, tc in zip(df[col].astype(str), df["tree_count"].astype(int), strict=False)
    }


def sample_weights_for_tree_oversample(
    paths: list[Path],
    tree_counts: dict[str, int],
    *,
    tree_weight: float = 1.0,
) -> list[float]:
    """Per-chip WeightedRandomSampler weights; tree_weight>1 upsamples register targets."""
    if tree_weight <= 1.0:
        return [1.0] * len(paths)
    out: list[float] = []
    for p in paths:
        n = tree_counts.get(p.stem, 0)
        out.append(float(tree_weight if n > 0 else 1.0))
    return out

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


def chip_origin_xy(chip_id: str) -> tuple[float, float] | None:
    """Origin corner from a chip id `gent_50m_{x}_{y}`."""
    parts = chip_id.split("_")
    if len(parts) < 2:
        return None
    try:
        return float(parts[-2]), float(parts[-1])
    except ValueError:
        return None


def order_paths_outward(
    paths: list[Path],
    *,
    origins: dict[str, tuple[float, float]] | None = None,
    chip_size_m: float = 50.0,
) -> tuple[list[Path], tuple[float, float] | None]:
    """Order chips so inference starts at the bbox center and grows outward.

    Returns the ordered paths and the center (x, y) in the same coordinates as the
    chip origins. Paths whose origin cannot be read stay at the end.
    """
    half = chip_size_m / 2.0
    placed: list[tuple[Path, float, float]] = []
    unknown: list[Path] = []
    for path in paths:
        origin = None if origins is None else origins.get(path.stem)
        if origin is None:
            origin = chip_origin_xy(path.stem)
        if origin is None:
            unknown.append(path)
            continue
        x0, y0 = origin
        placed.append((path, x0 + half, y0 + half))
    if not placed:
        return list(paths), None
    xs = [x for _, x, _ in placed]
    ys = [y for _, _, y in placed]
    center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    cx, cy = center
    placed.sort(key=lambda item: ((item[1] - cx) ** 2 + (item[2] - cy) ** 2, item[0].name))
    return [item[0] for item in placed] + unknown, center


def chip_paths_outward_from_index(
    index_parquet: Path,
) -> tuple[list[Path], tuple[float, float] | None]:
    """Existing chips from a manifest, nearest the coverage center first."""
    df = pd.read_parquet(index_parquet)
    col = "chip_id" if "chip_id" in df.columns else df.columns[0]
    origins: dict[str, tuple[float, float]] = {}
    paths: list[Path] = []
    has_origin = "origin_x" in df.columns and "origin_y" in df.columns
    for row in df.itertuples(index=False):
        cid = str(getattr(row, col))
        path = chips_dir() / f"{cid}.npz"
        if not path.is_file():
            continue
        paths.append(path)
        if has_origin:
            origins[cid] = (float(getattr(row, "origin_x")), float(getattr(row, "origin_y")))
    return order_paths_outward(paths, origins=origins or None)


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

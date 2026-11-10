"""Rewrite heatmap (variable σ) in existing chips without re-reading LAZ."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from treelidarnet.data.build_chips import _trees_in_chip
from treelidarnet.data.chips import CHIP_SIZE_M, heatmap_sigma_m, rasterize_heatmap
from treelidarnet.data.grb import GrbIndex, make_hard_negative_mask
from treelidarnet.paths import chips_dir, data_dir


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    # numpy appends .npz if missing; use a sibling *.tmp.npz then atomic replace.
    tmp = path.with_name(f"{path.stem}.tmp.npz")
    if tmp.is_file():
        tmp.unlink()
    out = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez_compressed(str(tmp), **out)
    if not tmp.is_file():
        raise OSError(f"temp chip not written: {tmp}")
    tmp.replace(path)


def _tree_targets(chip_trees: pd.DataFrame) -> list[tuple[float, float, float]]:
    return [
        (float(r["x"]), float(r["y"]), heatmap_sigma_m(r.get("stamomtrek")))
        for r in chip_trees.to_dict("records")
    ]


def patch_chips_heatmaps(
    *,
    trees_parquet: Path,
    grb_dir: Path | None = None,
    road_buffer_m: float = 2.0,
    refresh_hard_negative: bool = True,
    limit: int | None = None,
) -> int:
    trees_df = pd.read_parquet(trees_parquet)
    paths = sorted(chips_dir().glob("gent_50m_*.npz"))
    if limit is not None:
        paths = paths[:limit]
    grb: GrbIndex | None = None
    if refresh_hard_negative:
        grb_dir = grb_dir or (data_dir() / "raw" / "grb")
        if (Path(grb_dir) / "gent_grb_buildings.geojson").is_file():
            grb = GrbIndex.from_dir(Path(grb_dir), road_buffer_m=road_buffer_m)
            print(f"GRB refresh: {grb_dir}", flush=True)
        else:
            print("GRB refresh skipped (no geojson)", flush=True)
    patched = 0
    for i, path in enumerate(paths, 1):
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        x0 = float(arrays["origin"][0])
        y0 = float(arrays["origin"][1])
        chip_trees = _trees_in_chip(trees_df, x0, y0, CHIP_SIZE_M)
        tree_xy = _tree_targets(chip_trees)
        heat = rasterize_heatmap(tree_xy, origin_x=x0, origin_y=y0)
        arrays["heatmap"] = heat.astype(np.float32)
        if grb is not None:
            arrays["hard_negative"] = make_hard_negative_mask(
                grb, origin_x=x0, origin_y=y0, heatmap=heat
            )
        _atomic_save_npz(path, arrays)
        patched += 1
        if i % 500 == 0:
            print(f"patched heatmap {i}/{len(paths)}", flush=True)
    print(f"Done patching heatmap on {patched} chips", flush=True)
    return patched

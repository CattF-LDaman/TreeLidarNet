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
from treelidarnet.data.laz_io import crop_cloud, read_laz_cloud
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


def rebuild_empty_chips(
    *,
    trees_parquet: Path,
    laz_dir: Path,
    index_parquet: Path | None = None,
    dtm_path: Path | None = None,
    grb_dir: Path | None = None,
    road_buffer_m: float = 2.0,
    limit: int | None = None,
) -> int:
    """Rewrite chips whose stored cloud is empty, using every LAZ for that corner."""
    index_parquet = index_parquet or (manifests_dir() / "chips_index_full.parquet")
    laz_index = index_laz_files(laz_dir)
    if not laz_index:
        raise FileNotFoundError(f"No indexed LAZ in {laz_dir}")
    df = pd.read_parquet(index_parquet)
    empty = df[df["point_count"] <= 0]
    if limit:
        empty = empty.iloc[:limit]
    groups: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for row in empty.itertuples(index=False):
        tx = int(np.floor(float(row.origin_x) / 500.0) * 500)
        ty = int(np.floor(float(row.origin_y) / 500.0) * 500)
        groups.setdefault((tx, ty), []).append((float(row.origin_x), float(row.origin_y)))
    grb_index = _grb_index(grb_dir, road_buffer_m)
    trees_df = pd.read_parquet(trees_parquet)
    index_mtime = index_parquet.stat().st_mtime
    rebuilt: dict[str, dict] = {}
    filled = 0
    seen = 0
    total = int(len(empty))
    for (tx, ty), origins in groups.items():
        pending: list[tuple[float, float]] = []
        for x0, y0 in origins:
            fresh = _sidecar_if_newer(x0, y0, newer_than=index_mtime)
            if fresh is None:
                pending.append((x0, y0))
                continue
            seen += 1
            rebuilt[str(fresh["chip_id"])] = fresh
            if int(fresh.get("point_count") or 0) > 0:
                filled += 1
            if seen % 200 == 0 or seen == total:
                print(
                    f"rebuild empty {seen}/{total} filled={filled} tiles={len(groups)}",
                    flush=True,
                )
        if not pending:
            continue
        paths = laz_index.get((tx, ty), [])
        clouds = [_loaded_cloud(path) for path in paths]
        clouds = [cloud for cloud in clouds if cloud is not None]
        for x0, y0 in pending:
            seen += 1
            chunks_xyz = []
            chunks_feat = []
            for xyz, feat in clouds:
                cropped_xyz, cropped_feat = crop_cloud(
                    xyz, feat, xmin=x0, ymin=y0, xmax=x0 + CHIP_SIZE_M, ymax=y0 + CHIP_SIZE_M
                )
                if len(cropped_xyz):
                    chunks_xyz.append(cropped_xyz)
                    chunks_feat.append(cropped_feat)
            if chunks_xyz:
                cloud = (np.concatenate(chunks_xyz), np.concatenate(chunks_feat))
            else:
                cloud = (
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0, 3), dtype=np.float32),
                )
            _, meta = build_chip(
                x0=x0,
                y0=y0,
                laz_paths=[],
                trees_df=trees_df,
                dtm_path=dtm_path,
                grb_index=grb_index,
                cloud=cloud,
            )
            rebuilt[meta["chip_id"]] = meta
            if meta["point_count"] > 0:
                filled += 1
            if seen % 200 == 0 or seen == total:
                print(
                    f"rebuild empty {seen}/{total} filled={filled} tiles={len(groups)}",
                    flush=True,
                )
    if rebuilt:
        df = df.set_index("chip_id", drop=False)
        for cid, meta in rebuilt.items():
            if cid not in df.index:
                continue
            df.loc[cid, "point_count"] = meta["point_count"]
            df.loc[cid, "mean_pts_per_m2"] = meta["mean_pts_per_m2"]
            df.loc[cid, "tree_count"] = meta["tree_count"]
            df.at[cid, "qc"] = meta["qc"]
        df = df.reset_index(drop=True)
        df.to_parquet(index_parquet, index=False)
    print(f"Done rebuild: {len(rebuilt)} chips, {filled} now have points", flush=True)
    return filled


def _grb_index(grb_dir: Path | None, road_buffer_m: float) -> GrbIndex | None:
    if grb_dir is None:
        default_grb = data_dir() / "raw" / "grb"
        if (default_grb / "gent_grb_buildings.geojson").is_file():
            grb_dir = default_grb
    if grb_dir is None:
        return None
    print(f"GRB hard negatives: {grb_dir}", flush=True)
    return GrbIndex.from_dir(grb_dir, road_buffer_m=road_buffer_m)


def _sidecar_if_newer(x0: float, y0: float, *, newer_than: float) -> dict | None:
    """Meta written by a rebuild that has not been committed to the index yet."""
    path = chips_dir() / f"{chip_id(x0, y0)}.meta.json"
    if not path.is_file() or path.stat().st_mtime <= newer_than:
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(meta, dict) or "chip_id" not in meta:
        return None
    return meta


def _loaded_cloud(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            signature = handle.read(4)
        if signature != b"LASF":
            print(f"skip unreadable LAZ {path.name}: signature {signature!r}", flush=True)
            return None
        return read_laz_cloud(path)
    except Exception as exc:
        print(f"skip unreadable LAZ {path.name}: {exc}", flush=True)
        return None

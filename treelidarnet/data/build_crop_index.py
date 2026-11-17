"""Build crops_index.parquet for the PointNet++ verifier."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from treelidarnet.data.chips import chip_id
from treelidarnet.data.crops import (
    chip_origin_for_point,
    crop_row_meta,
    global_to_chip_local,
    hard_negative_cell_centers,
)
from treelidarnet.data.split import chip_split_for_sheet, load_split_meta
from treelidarnet.paths import chips_dir, manifests_dir


def _split_bucket(key: str) -> str:
    """Stable 85/10/5 train/val/test from any string id."""
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    bucket = h % 100
    if bucket < 85:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


def _split_for_tree(
    *,
    split_meta: dict | None,
    chip_id_value: str,
    gis_id: str,
) -> str:
    if split_meta is None:
        return "train"
    sheet = split_meta.get("tree_sheet") or {}
    sheet_id = sheet.get(gis_id)
    if sheet_id:
        name = chip_split_for_sheet(str(sheet_id), split_meta)
        if name:
            return name
    return _split_bucket(gis_id)


def build_crop_index(
    *,
    trees_parquet: Path,
    chips_index: Path | None = None,
    out_path: Path | None = None,
    split_meta_path: Path | None = None,
    positives_only: bool = False,
    grb_negatives_per_chip: int = 2,
    grb_max_chips: int = 15_000,
    limit: int | None = None,
    seed: int = 42,
) -> Path:
    """Write manifest of crop training rows."""
    out_path = out_path or (manifests_dir() / "crops_index.parquet")
    trees = pd.read_parquet(trees_parquet)
    trees = trees[trees["eligible"]]
    split_meta = None
    smp = split_meta_path or (manifests_dir() / "split_meta.json")
    if smp.is_file():
        split_meta = load_split_meta(smp)

    chip_point_count: dict[str, int] = {}
    if chips_index is not None and Path(chips_index).is_file():
        idx = pd.read_parquet(chips_index)
        if "chip_id" in idx.columns and "point_count" in idx.columns:
            chip_point_count = dict(zip(idx["chip_id"], idx["point_count"], strict=False))

    rows: list[dict] = []
    rng = np.random.default_rng(seed)

    for i, r in enumerate(trees.to_dict("records")):
        if limit is not None and len(rows) >= limit:
            break
        gx, gy = float(r["x"]), float(r["y"])
        x0, y0 = chip_origin_for_point(gx, gy)
        cid = chip_id(x0, y0)
        pc = chip_point_count.get(cid)
        if pc is not None and pc < 32:
            continue
        chip_path = chips_dir() / f"{cid}.npz"
        if not chip_path.is_file():
            continue
        lx, ly = global_to_chip_local(gx, gy, origin_x=x0, origin_y=y0)
        crop_id = f"reg_{r['gis_id']}"
        split = _split_for_tree(split_meta=split_meta, chip_id_value=cid, gis_id=str(r["gis_id"]))
        rows.append(
            crop_row_meta(
                crop_id=crop_id,
                label=1,
                source="register",
                chip_id_value=cid,
                center_x=lx,
                center_y=ly,
                split=split,
                gis_id=str(r["gis_id"]),
                origin_x=x0,
                origin_y=y0,
            )
        )

    if not positives_only:
        chip_ids: list[str] = []
        if chips_index is not None and Path(chips_index).is_file():
            idx = pd.read_parquet(chips_index)
            if "chip_id" in idx.columns:
                m = np.ones(len(idx), dtype=bool)
                if "point_count" in idx.columns:
                    m = idx["point_count"].fillna(0).astype(int) > 0
                chip_ids = idx.loc[m, "chip_id"].astype(str).tolist()
        if not chip_ids:
            chip_ids = sorted(p.stem for p in chips_dir().glob("gent_50m_*.npz"))
        if grb_max_chips > 0 and len(chip_ids) > grb_max_chips:
            pick = rng.choice(len(chip_ids), size=grb_max_chips, replace=False)
            chip_ids = [chip_ids[i] for i in sorted(pick)]
        chip_files = [chips_dir() / f"{cid}.npz" for cid in chip_ids]
        for chip_path in chip_files:
            if limit is not None and len(rows) >= limit:
                break
            cid = chip_path.stem
            if chip_point_count.get(cid, 1) == 0:
                continue
            data = np.load(chip_path)
            try:
                if "hard_negative" not in data:
                    continue
                hn = data["hard_negative"].astype(np.float32)
                centers = hard_negative_cell_centers(
                    hn, max_samples=grb_negatives_per_chip, rng=rng
                )
                origin = data["origin"].astype(np.float64)
                x0, y0 = float(origin[0]), float(origin[1])
            finally:
                data.close()
            for j, (lx, ly) in enumerate(centers):
                neg_id = f"grb_{cid}_{j}"
                rows.append(
                    crop_row_meta(
                        crop_id=neg_id,
                        label=0,
                        source="grb",
                        chip_id_value=cid,
                        center_x=lx,
                        center_y=ly,
                        split=_split_bucket(neg_id),
                        origin_x=x0,
                        origin_y=y0,
                    )
                )

    if limit is not None:
        rows = rows[:limit]

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    pos = int((df["label"] == 1).sum()) if len(df) else 0
    neg = int((df["label"] == 0).sum()) if len(df) else 0
    if len(df):
        for split_name in ("train", "val", "test"):
            sub = df[df["split"] == split_name]
            p = int((sub["label"] == 1).sum())
            n = int((sub["label"] == 0).sum())
            print(f"  {split_name}: {len(sub)} rows ({p} pos, {n} neg)", flush=True)
    print(f"Wrote {out_path} ({len(df)} rows: {pos} pos, {neg} neg)", flush=True)
    return out_path
"""Persist explorer false-positive flags for verifier retrain."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from treelidarnet.data.crs import wgs84_to_lambert
from treelidarnet.paths import manifests_dir

DEDUP_LONLAT_DEC = 7
VALID_REASONS = frozenset({"default", "road", "shrub", "duplicate", "other"})


def default_flags_path() -> Path:
    return manifests_dir() / "explorer_qa_flags.parquet"


def flag_id_for_lonlat(lon: float, lat: float) -> str:
    key = f"{round(lon, DEDUP_LONLAT_DEC)},{round(lat, DEDUP_LONLAT_DEC)}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"fp_{digest}"


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "flag_id",
            "lon",
            "lat",
            "x",
            "y",
            "bev",
            "v2",
            "v3",
            "dist_register_m",
            "reason",
            "note",
            "flagged_at",
            "bundle_source",
            "active",
            "gui_hidden",
        ]
    )


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_df()
    if "gui_hidden" not in df.columns:
        df = df.copy()
        df["gui_hidden"] = False
    if "active" not in df.columns:
        df = df.copy()
        df["active"] = True
    return df


def load_flags(
    path: Path | None = None,
    *,
    active_only: bool = True,
    gui_visible_only: bool = False,
) -> pd.DataFrame:
    path = path or default_flags_path()
    if not path.is_file():
        return _empty_df()
    df = _ensure_columns(pd.read_parquet(path))
    if active_only and len(df):
        df = df.loc[df["active"].astype(bool)]
    if gui_visible_only and len(df):
        df = df.loc[~df["gui_hidden"].astype(bool)]
    return df


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def upsert_flag(
    *,
    lon: float,
    lat: float,
    bev: float | None = None,
    v2: float | None = None,
    v3: float | None = None,
    dist_register_m: float | None = None,
    reason: str = "default",
    note: str = "",
    bundle_source: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    path = path or default_flags_path()
    reason = reason if reason in VALID_REASONS else "default"
    x, y = wgs84_to_lambert(lon, lat)
    fid = flag_id_for_lonlat(lon, lat)
    row = {
        "flag_id": fid,
        "lon": float(lon),
        "lat": float(lat),
        "x": float(x),
        "y": float(y),
        "bev": float(bev) if bev is not None else None,
        "v2": float(v2) if v2 is not None else None,
        "v3": float(v3) if v3 is not None else None,
        "dist_register_m": float(dist_register_m) if dist_register_m is not None else None,
        "reason": reason,
        "note": str(note or ""),
        "flagged_at": datetime.now(UTC).isoformat(),
        "bundle_source": str(bundle_source or ""),
        "active": True,
        "gui_hidden": False,
    }
    if path.is_file():
        df = _ensure_columns(pd.read_parquet(path))
    else:
        df = _empty_df()
    if len(df) and "flag_id" in df.columns:
        df = df.loc[df["flag_id"] != fid]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_df(df, path)
    return row


def clear_flag(flag_id: str, *, path: Path | None = None) -> bool:
    path = path or default_flags_path()
    if not path.is_file():
        return False
    df = _ensure_columns(pd.read_parquet(path))
    if df.empty or "flag_id" not in df.columns:
        return False
    mask = df["flag_id"] == flag_id
    if not mask.any():
        return False
    df.loc[mask, "active"] = False
    _save_df(df, path)
    return True


def mark_flags_gui_hidden(
    flag_ids: set[str] | list[str],
    *,
    path: Path | None = None,
) -> int:
    """Hide flags in the explorer UI after they are in the crop training index."""
    ids = {str(i) for i in flag_ids if i}
    if not ids:
        return 0
    path = path or default_flags_path()
    if not path.is_file():
        return 0
    df = _ensure_columns(pd.read_parquet(path))
    if df.empty:
        return 0
    mask = df["flag_id"].astype(str).isin(ids) & df["active"].astype(bool)
    if not mask.any():
        return 0
    df.loc[mask, "gui_hidden"] = True
    _save_df(df, path)
    return int(mask.sum())


def sync_gui_hidden_from_crop_index(
    crops_index_path: Path,
    *,
    qa_flags_path: Path | None = None,
) -> int:
    """Mark active QA flags that appear as explorer_fp crops as GUI-hidden."""
    if not crops_index_path.is_file():
        return 0
    ci = pd.read_parquet(crops_index_path)
    if ci.empty or "source" not in ci.columns or "crop_id" not in ci.columns:
        return 0
    fp_ids = set(ci.loc[ci["source"] == "explorer_fp", "crop_id"].astype(str))
    return mark_flags_gui_hidden(fp_ids, path=qa_flags_path)


def flags_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df.to_dict(orient="records")
    for row in out:
        for k, v in list(row.items()):
            if hasattr(v, "item"):
                row[k] = v.item()
            elif pd.isna(v):
                row[k] = None
    return out


def flags_in_bbox(
    df: pd.DataFrame,
    *,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> pd.DataFrame:
    if df.empty:
        return df
    m = (
        (df["lon"] >= min_lon)
        & (df["lon"] <= max_lon)
        & (df["lat"] >= min_lat)
        & (df["lat"] <= max_lat)
    )
    return df.loc[m]


def export_flags_snapshot(
    dest: Path,
    *,
    source_path: Path | None = None,
) -> int:
    df = load_flags(source_path, active_only=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return len(df)
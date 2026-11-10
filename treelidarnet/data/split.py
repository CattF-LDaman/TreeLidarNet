"""Train/val/test split metadata (NGI kaartblad lists)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def load_split_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_split_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def chip_split_for_sheet(sheet_id: str, meta: dict[str, Any]) -> str | None:
    sheets = meta.get("sheets") or {}
    for split_name in ("train", "val", "test"):
        if sheet_id in (sheets.get(split_name) or []):
            return split_name
    return None


def init_split_meta_from_example(example: Path, dest: Path) -> None:
    if dest.is_file():
        raise FileExistsError(f"Already exists: {dest}")
    shutil.copy(example, dest)


def update_gent_hashes(
    meta: dict[str, Any],
    *,
    sha256: str,
    record_count: int,
    eligible_count: int,
) -> dict[str, Any]:
    meta = dict(meta)
    meta["gent_trees_parquet_sha256"] = sha256
    meta["gent_record_count"] = record_count
    meta["gent_eligible_count"] = eligible_count
    return meta

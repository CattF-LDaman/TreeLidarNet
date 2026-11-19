"""Build columnar explorer bundle from dual-scored inference export."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
from scipy.spatial import cKDTree

from treelidarnet.data.crops import chip_id_for_point
from treelidarnet.data.crs import wgs84_to_lambert
from treelidarnet.data.rgb_chip_gate import chip_has_valid_rgb
from treelidarnet.export_mapjson import _tree_radius_m
from treelidarnet.media.showcase_select import load_verified_predictions
from treelidarnet.paths import manifests_dir, repo_root

BUNDLE_MAGIC = b"TLEX"
BUNDLE_VERSION = 1
BUNDLE_COLUMNS = ("lon", "lat", "bev", "v2", "v3", "dist_register_m", "has_rgb")
DETECTION_DTYPE = np.dtype(
    [
        ("lon", "<f4"),
        ("lat", "<f4"),
        ("bev", "<f4"),
        ("v2", "<f4"),
        ("v3", "<f4"),
        ("dist_register_m", "<f4"),
        ("has_rgb", "<f4"),
    ]
)
REGISTER_DTYPE = np.dtype(
    [
        ("lon", "<f4"),
        ("lat", "<f4"),
        ("radius_m", "<f4"),
        ("has_size", "<f4"),
    ]
)


@dataclass
class BuildBundleConfig:
    input_path: Path
    output_dir: Path
    trees_parquet: Path
    mark_rgb: bool = False
    bev_min_export: float | None = None


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".json" or path.name.endswith(".json"):
        return list(orjson.loads(path.read_bytes()).get("predictions") or [])
    return load_verified_predictions(path)


def _row_scores(row: dict[str, Any]) -> tuple[float, float, float]:
    bev = float(row.get("score") or 0.0)
    vs = row.get("verifier_scores") or {}
    v2 = float(vs.get("v2", row.get("verifier_score") or 0.0))
    v3 = float(vs.get("v3", row.get("verifier_score") or 0.0))
    if "verifier_scores" not in row and row.get("verifier_score") is not None:
        v2 = v3 = float(row["verifier_score"])
    return bev, v2, v3


def _load_register_arrays(trees_parquet: Path) -> tuple[np.ndarray, np.ndarray]:
    if not trees_parquet.is_file():
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=REGISTER_DTYPE)
    df = pd.read_parquet(trees_parquet)
    if df.empty or "eligible" not in df.columns:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=REGISTER_DTYPE)
    sub = df.loc[df["eligible"]]
    xy = sub[["x", "y"]].to_numpy(dtype=np.float64)
    reg = np.zeros(len(sub), dtype=REGISTER_DTYPE)
    for i, row in enumerate(sub.to_dict("records")):
        reg["lon"][i] = float(row["lon"])
        reg["lat"][i] = float(row["lat"])
        radius, has_size = _tree_radius_m(row.get("stamomtrek"))
        reg["radius_m"][i] = radius
        reg["has_size"][i] = 1.0 if has_size else 0.0
    return xy, reg


def build_explorer_bundle(cfg: BuildBundleConfig) -> dict[str, Any]:
    rows = _read_predictions(cfg.input_path)
    if cfg.bev_min_export is not None:
        rows = [r for r in rows if float(r.get("score") or 0) >= cfg.bev_min_export]

    tree_xy, register = _load_register_arrays(cfg.trees_parquet)
    tree_kd = cKDTree(tree_xy) if len(tree_xy) else None

    n = len(rows)
    detections = np.zeros(n, dtype=DETECTION_DTYPE)
    rgb_checked = 0
    for i, row in enumerate(rows):
        pos = row.get("position") or [0.0, 0.0]
        lon, lat = float(pos[0]), float(pos[1])
        bev, v2, v3 = _row_scores(row)
        x, y = wgs84_to_lambert(lon, lat)
        dist = float(tree_kd.query([x, y])[0]) if tree_kd is not None else 9999.0
        has_rgb = 0.0
        if cfg.mark_rgb:
            cid = chip_id_for_point(x, y)
            has_rgb = 1.0 if chip_has_valid_rgb(cid) else 0.0
            rgb_checked += 1
        detections["lon"][i] = lon
        detections["lat"][i] = lat
        detections["bev"][i] = bev
        detections["v2"][i] = v2
        detections["v3"][i] = v3
        detections["dist_register_m"][i] = dist
        detections["has_rgb"][i] = has_rgb

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    det_path = out / "detections.bin"
    reg_path = out / "register.bin"
    header = BUNDLE_MAGIC + struct.pack("<II", BUNDLE_VERSION, n)
    det_path.write_bytes(header + detections.tobytes())
    reg_header = BUNDLE_MAGIC + struct.pack("<II", BUNDLE_VERSION, len(register))
    reg_path.write_bytes(reg_header + register.tobytes())

    meta = {
        "version": BUNDLE_VERSION,
        "magic": BUNDLE_MAGIC.decode("ascii"),
        "columns": list(BUNDLE_COLUMNS),
        "detection_count": n,
        "register_count": int(len(register)),
        "source": str(cfg.input_path.resolve()),
        "trees_parquet": str(cfg.trees_parquet.resolve()),
        "built_at": datetime.now(UTC).isoformat(),
        "scores_precomputed": True,
        "mark_rgb": cfg.mark_rgb,
        "rgb_checked": rgb_checked,
        "detection_dtype_bytes": int(DETECTION_DTYPE.itemsize),
        "register_dtype_bytes": int(REGISTER_DTYPE.itemsize),
    }
    (out / "meta.json").write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    return meta


def default_bundle_paths() -> tuple[Path, Path]:
    """Preferred input (full verify gz, else live JSON) and output dir."""
    inf = repo_root() / "exports" / "inference"
    full = inf / "gent_city_gent_canopy_v2_verified_v2_v3_full.json.gz"
    live = inf / "live" / "verified_predictions.json"
    if full.is_file():
        return full, inf / "explorer"
    return live, inf / "explorer"


def cli_build_bundle(
    *,
    input_path: Path | None,
    output_dir: Path | None,
    trees_parquet: Path | None,
    mark_rgb: bool,
) -> None:
    default_in, default_out = default_bundle_paths()
    cfg = BuildBundleConfig(
        input_path=input_path or default_in,
        output_dir=output_dir or default_out,
        trees_parquet=trees_parquet or (manifests_dir() / "gent_trees.parquet"),
        mark_rgb=mark_rgb,
    )
    meta = build_explorer_bundle(cfg)
    print(
        f"Explorer bundle: {meta['detection_count']} detections, "
        f"{meta['register_count']} register → {cfg.output_dir}",
        flush=True,
    )
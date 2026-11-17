"""Stake-centered LiDAR crops from chip npz files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from treelidarnet.data.chips import CELL_SIZE_M, CHIP_SIZE_M, chip_id
from treelidarnet.paths import chips_dir

DEFAULT_CROP_RADIUS_M = 8.0
DEFAULT_NUM_POINTS = 512
DEFAULT_MIN_POINTS = 32
IN_CHANNELS = 6
IN_CHANNELS_RGB = 9


def crop_in_channels(*, use_rgb: bool = False) -> int:
    return IN_CHANNELS_RGB if use_rgb else IN_CHANNELS


def chip_origin_for_point(x: float, y: float) -> tuple[float, float]:
    """Lower-left Lambert origin of the 50 m chip containing (x, y)."""
    x0 = float(np.floor(x / CHIP_SIZE_M) * CHIP_SIZE_M)
    y0 = float(np.floor(y / CHIP_SIZE_M) * CHIP_SIZE_M)
    return x0, y0


def chip_id_for_point(x: float, y: float) -> str:
    x0, y0 = chip_origin_for_point(x, y)
    return chip_id(x0, y0)


def global_to_chip_local(
    x: float, y: float, *, origin_x: float, origin_y: float
) -> tuple[float, float]:
    return x - origin_x, y - origin_y


def normalize_point_features(feat: np.ndarray) -> np.ndarray:
    """Match ChipDataset BEV normalization."""
    out = feat.astype(np.float32, copy=True)
    if out.size:
        out[:, 0] = out[:, 0] / 65535.0
        out[:, 1] = out[:, 1] / 7.0
        out[:, 2] = out[:, 2] / 31.0
    return out


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32, copy=True)
    if out.size:
        out /= 65535.0
    return out


def build_crop_tensor(
    xyz: np.ndarray,
    feat: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    radius_m: float = DEFAULT_CROP_RADIUS_M,
    num_points: int = DEFAULT_NUM_POINTS,
    min_points: int = DEFAULT_MIN_POINTS,
    rng: np.random.Generator | None = None,
    rgb: np.ndarray | None = None,
    in_channels: int = IN_CHANNELS,
) -> np.ndarray | None:
    """Return (num_points, in_channels) centered on crop, or None if too sparse."""
    use_rgb = in_channels > IN_CHANNELS
    if use_rgb and (rgb is None or rgb.shape[0] != xyz.shape[0]):
        return None
    if xyz.shape[0] < min_points:
        return None
    dx = xyz[:, 0] - center_x
    dy = xyz[:, 1] - center_y
    dist2 = dx * dx + dy * dy
    keep = dist2 <= radius_m * radius_m
    if int(keep.sum()) < min_points:
        return None
    xyz_k = xyz[keep]
    feat_k = normalize_point_features(feat[keep])
    cx = float(center_x)
    cy = float(center_y)
    pts = np.zeros((len(xyz_k), in_channels), dtype=np.float32)
    pts[:, 0] = xyz_k[:, 0] - cx
    pts[:, 1] = xyz_k[:, 1] - cy
    pts[:, 2] = xyz_k[:, 2]
    pts[:, 3:6] = feat_k
    if use_rgb:
        pts[:, 6:9] = normalize_rgb(rgb[keep])
    rng = rng or np.random.default_rng()
    n = len(pts)
    if n >= num_points:
        idx = rng.choice(n, size=num_points, replace=False)
    else:
        idx = rng.choice(n, size=num_points, replace=True)
    return pts[idx]


def extract_crop_from_chip_path(
    chip_path: Path,
    *,
    center_x: float,
    center_y: float,
    radius_m: float = DEFAULT_CROP_RADIUS_M,
    num_points: int = DEFAULT_NUM_POINTS,
    min_points: int = DEFAULT_MIN_POINTS,
    rng: np.random.Generator | None = None,
    in_channels: int = IN_CHANNELS,
) -> np.ndarray | None:
    """Load npz and extract a crop; center in chip-local meters."""
    if not chip_path.is_file():
        return None
    data = np.load(chip_path)
    try:
        xyz = data["xyz"].astype(np.float32)
        feat = data["feat"].astype(np.float32)
        rgb = data["rgb"].astype(np.float32) if in_channels > IN_CHANNELS and "rgb" in data else None
        return build_crop_tensor(
            xyz,
            feat,
            center_x=center_x,
            center_y=center_y,
            radius_m=radius_m,
            num_points=num_points,
            min_points=min_points,
            rng=rng,
            rgb=rgb,
            in_channels=in_channels,
        )
    finally:
        data.close()


def extract_crop_at_global(
    *,
    global_x: float,
    global_y: float,
    chip_path: Path | None = None,
    radius_m: float = DEFAULT_CROP_RADIUS_M,
    num_points: int = DEFAULT_NUM_POINTS,
    min_points: int = DEFAULT_MIN_POINTS,
    rng: np.random.Generator | None = None,
    in_channels: int = IN_CHANNELS,
) -> np.ndarray | None:
    """Crop around a Lambert stake or detection; resolves chip if path omitted."""
    x0, y0 = chip_origin_for_point(global_x, global_y)
    local_x, local_y = global_to_chip_local(global_x, global_y, origin_x=x0, origin_y=y0)
    path = chip_path or (chips_dir() / f"{chip_id(x0, y0)}.npz")
    return extract_crop_from_chip_path(
        path,
        center_x=local_x,
        center_y=local_y,
        radius_m=radius_m,
        num_points=num_points,
        min_points=min_points,
        rng=rng,
        in_channels=in_channels,
    )


def hard_negative_cell_centers(
    hard_negative: np.ndarray,
    *,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    threshold: float = 0.5,
    max_samples: int = 8,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float]]:
    """Random local xy centers on GRB hard-negative cells."""
    mask = hard_negative >= threshold
    if not np.any(mask):
        return []
    ys, xs = np.nonzero(mask)
    rng = rng or np.random.default_rng()
    if len(xs) > max_samples:
        pick = rng.choice(len(xs), size=max_samples, replace=False)
        ys, xs = ys[pick], xs[pick]
    out: list[tuple[float, float]] = []
    for py, px in zip(ys, xs, strict=False):
        lx = (px + 0.5) * CELL_SIZE_M
        ly = (py + 0.5) * CELL_SIZE_M
        out.append((lx, ly))
    return out


def crop_row_meta(
    *,
    crop_id: str,
    label: int,
    source: str,
    chip_id_value: str,
    center_x: float,
    center_y: float,
    split: str,
    gis_id: str | None = None,
    origin_x: float | None = None,
    origin_y: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "crop_id": crop_id,
        "label": int(label),
        "source": source,
        "chip_id": chip_id_value,
        "center_x": float(center_x),
        "center_y": float(center_y),
        "split": split,
    }
    if gis_id is not None:
        row["gis_id"] = gis_id
    if origin_x is not None:
        row["origin_x"] = float(origin_x)
        row["origin_y"] = float(origin_y)
    return row
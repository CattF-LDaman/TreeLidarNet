"""Read and crop LAZ tiles."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np


def _las_has_rgb(las: laspy.LasData) -> bool:
    return all(hasattr(las, name) for name in ("red", "green", "blue"))


def _header_has_rgb(header: laspy.LasHeader) -> bool:
    names = set(header.point_format.dimension_names)
    return {"red", "green", "blue"}.issubset(names)


def laz_file_readable(path: Path) -> bool:
    """True when path looks like a normal LAS/LAZ (not COPC sidecar garbage)."""
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"LASF"
    except OSError:
        return False


_CHUNK_POINTS = 400_000


def _append_bbox_from_laz(
    path: Path,
    *,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    include_rgb: bool,
    want_rgb: bool,
    xyz_parts: list[np.ndarray],
    feat_parts: list[np.ndarray],
    rgb_parts: list[np.ndarray],
) -> bool:
    """Stream one LAZ in chunks; append only points inside bbox. Returns whether RGB was seen."""
    if not laz_file_readable(path):
        return want_rgb
    try:
        with laspy.open(path) as f:
            file_has_rgb = include_rgb and want_rgb and _header_has_rgb(f.header)
            for chunk in f.chunk_iterator(_CHUNK_POINTS):
                x = np.asarray(chunk.x, dtype=np.float64)
                y = np.asarray(chunk.y, dtype=np.float64)
                z = np.asarray(chunk.z, dtype=np.float64)
                m = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
                if not np.any(m):
                    continue
                xyz_parts.append(
                    np.stack([x[m], y[m], z[m]], axis=1).astype(np.float32, copy=False)
                )
                feat_parts.append(
                    np.stack(
                        [
                            np.asarray(chunk.intensity[m], dtype=np.float32),
                            np.asarray(chunk.return_number[m], dtype=np.float32),
                            np.asarray(chunk.classification[m], dtype=np.float32),
                        ],
                        axis=1,
                    )
                )
                if file_has_rgb:
                    rgb_parts.append(
                        np.stack(
                            [
                                np.asarray(chunk.red[m], dtype=np.float32),
                                np.asarray(chunk.green[m], dtype=np.float32),
                                np.asarray(chunk.blue[m], dtype=np.float32),
                            ],
                            axis=1,
                        )
                    )
                elif include_rgb and want_rgb:
                    want_rgb = False
    except Exception:
        return want_rgb
    return want_rgb


def read_laz_bbox(
    paths: list[Path],
    *,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    dedupe_tol: float = 0.01,
    include_rgb: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xyz (N,3) Lambert+TAW and features (N,F): intensity, return_number, class.

    When ``include_rgb`` is True and tiles expose RGB, also return (N,3) uint16 RGB.
    Uses chunked LAZ reads so full tiles are not held in memory at once.
    """
    xyz_parts: list[np.ndarray] = []
    feat_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    want_rgb = include_rgb
    for path in paths:
        want_rgb = _append_bbox_from_laz(
            path,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            include_rgb=include_rgb,
            want_rgb=want_rgb,
            xyz_parts=xyz_parts,
            feat_parts=feat_parts,
            rgb_parts=rgb_parts,
        )
    if not xyz_parts:
        empty_xyz = np.zeros((0, 3), dtype=np.float32)
        empty_feat = np.zeros((0, 3), dtype=np.float32)
        if include_rgb:
            return empty_xyz, empty_feat, np.zeros((0, 3), dtype=np.float32)
        return empty_xyz, empty_feat
    xyz = np.concatenate(xyz_parts, axis=0)
    feat = np.concatenate(feat_parts, axis=0)
    del xyz_parts, feat_parts
    if include_rgb and rgb_parts:
        rgb = np.concatenate(rgb_parts, axis=0)
        del rgb_parts
        if len(rgb) == len(xyz):
            xyz, feat, rgb = dedupe_points(xyz, feat, rgb=rgb, tol=dedupe_tol)
            return xyz.astype(np.float32), feat.astype(np.float32), rgb.astype(np.float32)
    xyz, feat = dedupe_points(xyz, feat, tol=dedupe_tol)
    if include_rgb:
        return xyz.astype(np.float32), feat.astype(np.float32), np.zeros(
            (len(xyz), 3), dtype=np.float32
        )
    return xyz.astype(np.float32), feat.astype(np.float32)


def read_laz_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """All points in one LAZ. xyz is float64 Lambert; features match read_laz_bbox."""
    las = laspy.read(path)
    xyz = np.stack(
        [
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        ],
        axis=1,
    )
    feat = np.stack(
        [
            np.asarray(las.intensity, dtype=np.float32),
            np.asarray(las.return_number, dtype=np.float32),
            np.asarray(las.classification, dtype=np.float32),
        ],
        axis=1,
    )
    return xyz, feat


def crop_cloud(
    xyz: np.ndarray,
    feat: np.ndarray,
    *,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) == 0:
        return xyz, feat
    m = (xyz[:, 0] >= xmin) & (xyz[:, 0] < xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] < ymax)
    return xyz[m], feat[m]


def dedupe_points(
    xyz: np.ndarray,
    feat: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    tol: float = 0.01,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(xyz) == 0:
        if rgb is not None:
            return xyz, feat, rgb
        return xyz, feat
    keys = np.round(xyz / tol).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    if rgb is not None:
        return xyz[idx], feat[idx], rgb[idx]
    return xyz[idx], feat[idx]
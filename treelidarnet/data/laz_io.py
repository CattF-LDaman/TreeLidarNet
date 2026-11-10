"""Read and crop LAZ tiles."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np


def read_laz_bbox(
    paths: list[Path],
    *,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    dedupe_tol: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Return xyz (N,3) Lambert+TAW and features (N,F): intensity, return_number, class."""
    chunks_xyz: list[np.ndarray] = []
    chunks_feat: list[np.ndarray] = []
    for path in paths:
        if not path.is_file():
            continue
        las = laspy.read(path)
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        z = np.asarray(las.z, dtype=np.float64)
        m = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
        if not np.any(m):
            continue
        xyz = np.stack([x[m], y[m], z[m]], axis=1)
        intensity = np.asarray(las.intensity[m], dtype=np.float32)
        ret = np.asarray(las.return_number[m], dtype=np.float32)
        klass = np.asarray(las.classification[m], dtype=np.float32)
        feat = np.stack([intensity, ret, klass], axis=1)
        chunks_xyz.append(xyz)
        chunks_feat.append(feat)
    if not chunks_xyz:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    xyz = np.concatenate(chunks_xyz, axis=0)
    feat = np.concatenate(chunks_feat, axis=0)
    xyz, feat = dedupe_points(xyz, feat, tol=dedupe_tol)
    return xyz.astype(np.float32), feat.astype(np.float32)


def dedupe_points(
    xyz: np.ndarray,
    feat: np.ndarray,
    *,
    tol: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) == 0:
        return xyz, feat
    keys = np.round(xyz / tol).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    return xyz[idx], feat[idx]

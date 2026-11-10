"""Height above ground (SPEC §5.1 fallback chain)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol


def sample_dtm_at_points(
    dtm_path: Path,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Sample DTM elevations at Lambert xy; nodata → nan."""
    ground = np.full(len(x), np.nan, dtype=np.float64)
    if not dtm_path.is_file():
        return ground
    with rasterio.open(dtm_path) as src:
        for i in range(len(x)):
            try:
                r, c = rowcol(src.transform, x[i], y[i])
            except Exception:
                continue
            if 0 <= r < src.height and 0 <= c < src.width:
                val = src.read(1)[r, c]
                if val != src.nodata:
                    ground[i] = float(val)
    return ground


def height_above_ground(
    xyz: np.ndarray,
    feat: np.ndarray,
    *,
    dtm_path: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute h_agl per point; returns h_agl and QC dict."""
    z = xyz[:, 2].astype(np.float64)
    x, y = xyz[:, 0], xyz[:, 1]
    klass = feat[:, 2] if feat.shape[1] >= 3 else np.zeros(len(xyz))
    qc: dict = {"ground_class2_fraction": 0.0, "ground_unreliable": False}

    ground = np.full(len(xyz), np.nan, dtype=np.float64)
    if dtm_path:
        ground = sample_dtm_at_points(dtm_path, x, y)

    class2 = klass == 2
    qc["ground_class2_fraction"] = float(class2.mean()) if len(klass) else 0.0

    if np.any(class2):
        # per-point: use class2 median in 2m window would be expensive; use global median offset
        ground[class2] = z[class2]

    still_nan = np.isnan(ground)
    if np.any(still_nan):
        ground[still_nan] = np.nanmin(z) if len(z) else 0.0
        h_agl = z - ground
        qc["ground_unreliable"] = float(still_nan.mean()) > 0.1
    else:
        h_agl = z - ground

    return h_agl.astype(np.float32), qc

"""CHM local maxima baseline (SPEC Appendix A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.ndimage import maximum_filter

from treelidarnet.postprocess import Detection, nms_detections

BASELINE_VERSION = "chm_v1"
MIN_CANOPY_M = 3.0
PEAK_WINDOW_M = 5.0
NMS_RADIUS_M = 3.0


def chm_from_rasters(dsm_path: Path, dtm_path: Path) -> tuple[np.ndarray, rasterio.Affine, object]:
    with rasterio.open(dsm_path) as dsm, rasterio.open(dtm_path) as dtm:
        dsm_a = dsm.read(1).astype(np.float64)
        dtm_a = dtm.read(1).astype(np.float64)
        nodata = dsm.nodata if dsm.nodata is not None else -9999
        mask = (dsm_a != nodata) & (dtm_a != dtm.nodata if dtm.nodata is not None else True)
        chm = np.clip(dsm_a - dtm_a, 0, None)
        chm[~mask] = 0
        return chm.astype(np.float32), dsm.transform, dsm.crs


def peaks_from_chm(
    chm: np.ndarray,
    transform: rasterio.Affine,
    *,
    res_m: float = 1.0,
    min_height_m: float = MIN_CANOPY_M,
    window_m: float = PEAK_WINDOW_M,
) -> list[Detection]:
    window_px = max(1, int(round(window_m / res_m)))
    size = window_px * 2 + 1
    mx = maximum_filter(chm, size=size, mode="constant")
    mask = (chm >= min_height_m) & (chm >= mx - 1e-6)
    ys, xs = np.nonzero(mask)
    dets: list[Detection] = []
    for y, x in zip(ys, xs, strict=False):
        h = chm[y, x]
        xmap, ymap = xy(transform, y, x)
        dets.append(Detection(x=float(xmap), y=float(ymap), score=float(h) / 10.0))
    return nms_detections(dets, radius_m=NMS_RADIUS_M)


def chm_peaks_for_paths(dsm_path: Path, dtm_path: Path) -> list[Detection]:
    chm, transform, _ = chm_from_rasters(dsm_path, dtm_path)
    res = abs(transform.a)
    return peaks_from_chm(chm, transform, res_m=res)

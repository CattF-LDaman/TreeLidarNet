"""Heatmap peaks and NMS (docs/EVAL.md)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    score: float
    source_chip_ids: tuple[str, ...] = ()


def extract_peaks_from_heatmap(
    heatmap: np.ndarray,
    *,
    origin_x: float,
    origin_y: float,
    cell_size: float = 0.25,
    local_max_radius_m: float = 2.0,
    score_threshold: float = 0.35,
    top_k: int = 64,
) -> list[Detection]:
    h, w = heatmap.shape
    radius_px = max(1, int(round(local_max_radius_m / cell_size)))
    size = radius_px * 2 + 1
    mx = maximum_filter(heatmap, size=size, mode="constant")
    mask = (heatmap >= score_threshold) & (heatmap >= mx - 1e-9)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    scores = heatmap[ys, xs]
    order = np.argsort(-scores)
    dets: list[Detection] = []
    for idx in order[:top_k]:
        px, py = int(xs[idx]), int(ys[idx])
        x = origin_x + (px + 0.5) * cell_size
        y = origin_y + (py + 0.5) * cell_size
        dets.append(Detection(x=x, y=y, score=float(scores[idx])))
    return nms_detections(dets, radius_m=2.0)


def nms_detections(dets: list[Detection], *, radius_m: float) -> list[Detection]:
    if not dets:
        return []
    sorted_d = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    r2 = radius_m * radius_m
    for cand in sorted_d:
        ok = True
        for k in kept:
            dx = cand.x - k.x
            dy = cand.y - k.y
            if dx * dx + dy * dy <= r2:
                ok = False
                break
        if ok:
            kept.append(cand)
    return kept


def merge_chip_detections(
    chip_detections: list[tuple[str, list[Detection]]],
    *,
    radius_m: float = 2.5,
) -> list[Detection]:
    all_d: list[Detection] = []
    for chip_id, dets in chip_detections:
        for d in dets:
            all_d.append(
                Detection(
                    x=d.x,
                    y=d.y,
                    score=d.score,
                    source_chip_ids=(chip_id,) + d.source_chip_ids,
                )
            )
    sorted_d = sorted(all_d, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    r2 = radius_m * radius_m
    for cand in sorted_d:
        ok = True
        for k in kept:
            dx = cand.x - k.x
            dy = cand.y - k.y
            if dx * dx + dy * dy <= r2:
                chips = tuple(dict.fromkeys(k.source_chip_ids + cand.source_chip_ids))
                if cand.score > k.score:
                    kept.remove(k)
                    kept.append(
                        Detection(x=cand.x, y=cand.y, score=cand.score, source_chip_ids=chips)
                    )
                ok = False
                break
        if ok:
            kept.append(cand)
    return kept

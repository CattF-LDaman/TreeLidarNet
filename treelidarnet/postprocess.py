"""Heatmap peaks and NMS (docs/EVAL.md)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

# Minimum heatmap confidence (0–1) to export a peak at inference time.
DEFAULT_PEAK_SCORE_THRESHOLD = 0.20


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    score: float
    source_chip_ids: tuple[str, ...] = ()
    verifier_score: float | None = None


def extract_peaks_from_heatmap(
    heatmap: np.ndarray,
    *,
    origin_x: float,
    origin_y: float,
    cell_size: float = 0.25,
    local_max_radius_m: float = 2.0,
    score_threshold: float = DEFAULT_PEAK_SCORE_THRESHOLD,
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


def _nms_merge_detections_sorted(
    sorted_d: list[Detection],
    *,
    radius_m: float = 2.5,
) -> list[Detection]:
    # Same rule as a score-ordered scan: the first kept point within radius wins.
    # Buckets only limit which points are tested.
    cell = radius_m if radius_m > 0 else 1.0
    r2 = radius_m * radius_m
    buckets: dict[tuple[int, int], list[Detection]] = {}
    order: dict[int, int] = {}
    seq = 0

    def _cell(x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / cell), math.floor(y / cell))

    def _add(det: Detection) -> None:
        nonlocal seq
        buckets.setdefault(_cell(det.x, det.y), []).append(det)
        order[id(det)] = seq
        seq += 1

    def _drop(det: Detection) -> None:
        key = _cell(det.x, det.y)
        buckets[key].remove(det)
        del order[id(det)]

    for cand in sorted_d:
        ix, iy = _cell(cand.x, cand.y)
        hit: Detection | None = None
        hit_seq = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for kept in buckets.get((ix + dx, iy + dy), ()):
                    ddx = cand.x - kept.x
                    ddy = cand.y - kept.y
                    if ddx * ddx + ddy * ddy <= r2:
                        kept_seq = order[id(kept)]
                        if hit is None or kept_seq < hit_seq:
                            hit = kept
                            hit_seq = kept_seq
        if hit is None:
            _add(cand)
            continue
        if cand.score > hit.score:
            chips = tuple(dict.fromkeys(hit.source_chip_ids + cand.source_chip_ids))
            _drop(hit)
            vs = cand.verifier_score if cand.verifier_score is not None else hit.verifier_score
            _add(
                Detection(
                    x=cand.x,
                    y=cand.y,
                    score=cand.score,
                    source_chip_ids=chips,
                    verifier_score=vs,
                )
            )
    flat = [det for group in buckets.values() for det in group]
    flat.sort(key=lambda det: order[id(det)])
    return flat


def merge_detection_lists(
    *groups: list[Detection],
    radius_m: float = 2.5,
) -> list[Detection]:
    """Cross-chip NMS on one or more flat detection lists (score-ordered merge)."""
    all_d: list[Detection] = []
    for group in groups:
        all_d.extend(group)
    if not all_d:
        return []
    sorted_d = sorted(all_d, key=lambda d: d.score, reverse=True)
    return _nms_merge_detections_sorted(sorted_d, radius_m=radius_m)


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
                    verifier_score=d.verifier_score,
                )
            )
    if not all_d:
        return []
    sorted_d = sorted(all_d, key=lambda d: d.score, reverse=True)
    return _nms_merge_detections_sorted(sorted_d, radius_m=radius_m)

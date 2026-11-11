"""Match predictions to inventory; F1 and discoveries (docs/EVAL.md)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from treelidarnet.postprocess import Detection


@dataclass(frozen=True)
class GroundTruth:
    gis_id: str
    x: float
    y: float


@dataclass
class MatchResult:
    """Strict metrics treat every unmatched pred as FP (benchmark vs register)."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    matched_mean_dist_m: float
    discoveries: int

    @property
    def fp_audit(self) -> int:
        """FP excluding distant discoveries (inventory-gap candidates)."""
        return self.fp - self.discoveries

    @property
    def precision_audit(self) -> float:
        denom = self.tp + self.fp_audit
        return self.tp / denom if denom else 0.0

    @property
    def recall_audit(self) -> float:
        return self.recall

    @property
    def f1_audit(self) -> float:
        p, r = self.precision_audit, self.recall_audit
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _dist2(a: Detection, b: GroundTruth) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def match_detections(
    preds: list[Detection],
    gts: list[GroundTruth],
    *,
    match_distance_m: float = 3.0,
    discovery_distance_m: float = 5.0,
) -> MatchResult:
    match_r2 = match_distance_m * match_distance_m
    disc_r2 = discovery_distance_m * discovery_distance_m
    sorted_preds = sorted(preds, key=lambda p: p.score, reverse=True)
    gt_matched = [False] * len(gts)
    tp = 0
    dists: list[float] = []
    fp = 0
    discoveries = 0
    # Cell is the discovery radius, so a 3 m match and a 5 m near-miss are both
    # inside the 3×3 neighborhood. Same nearest-unmatched rule as a full scan.
    cell = discovery_distance_m if discovery_distance_m > 0 else 1.0
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, gt in enumerate(gts):
        key = (math.floor(gt.x / cell), math.floor(gt.y / cell))
        buckets.setdefault(key, []).append(i)

    def _near(pred: Detection, span: int, limit2: float, unmatched_only: bool) -> tuple[int, float]:
        ix = math.floor(pred.x / cell)
        iy = math.floor(pred.y / cell)
        best_i = -1
        best_d2 = limit2 + 1.0
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                for i in buckets.get((ix + dx, iy + dy), ()):
                    if unmatched_only and gt_matched[i]:
                        continue
                    d2 = _dist2(pred, gts[i])
                    if d2 <= limit2 and d2 < best_d2:
                        best_d2 = d2
                        best_i = i
        return best_i, best_d2

    match_span = max(1, math.ceil(match_distance_m / cell))
    for pred in sorted_preds:
        best_i, best_d2 = _near(pred, match_span, match_r2, True)
        if best_i >= 0:
            gt_matched[best_i] = True
            tp += 1
            dists.append(float(np.sqrt(best_d2)))
        else:
            near_i, _near_d2 = _near(pred, 1, disc_r2, False)
            if near_i >= 0:
                fp += 1
            else:
                discoveries += 1
                fp += 1

    fn = sum(1 for m in gt_matched if not m)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    mean_d = float(np.mean(dists)) if dists else 0.0
    return MatchResult(
        precision=prec,
        recall=rec,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        matched_mean_dist_m=mean_d,
        discoveries=discoveries,
    )


def discovery_candidates(
    preds: list[Detection],
    gts: list[GroundTruth],
    *,
    discovery_distance_m: float = 5.0,
) -> list[Detection]:
    """Predictions farther than discovery_distance_m from every GT."""
    disc_r2 = discovery_distance_m * discovery_distance_m
    out: list[Detection] = []
    for pred in preds:
        if not any(_dist2(pred, gt) <= disc_r2 for gt in gts):
            out.append(pred)
    return out


def bbox_area_km2(preds: list[Detection], *, padding_m: float = 100.0) -> float:
    if not preds:
        return 0.0
    xs = [p.x for p in preds]
    ys = [p.y for p in preds]
    w = max(xs) - min(xs) + 2 * padding_m
    h = max(ys) - min(ys) + 2 * padding_m
    return (w * h) / 1e6

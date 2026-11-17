"""Filter a BEV export through the crop verifier and report QA metrics."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from treelidarnet.data.crs import wgs84_to_lambert
from treelidarnet.eval import load_ground_truths
from treelidarnet.live_export import detections_to_rows
from treelidarnet.postprocess import Detection
from treelidarnet.verifier import score_detections


@dataclass
class FilterExportResult:
    input_count: int
    kept_count: int
    discovery_input: int
    discovery_kept: int
    discovery_retention: float
    register_near_input: int
    register_near_kept: int
    register_retention: float


def _load_predictions(path: Path) -> list[Detection]:
    with gzip.open(path, "rb") as handle:
        payload = json.loads(handle.read())
    out: list[Detection] = []
    for row in payload.get("predictions", []):
        lon, lat = row["position"]
        x, y = wgs84_to_lambert(lon, lat)
        out.append(Detection(x=x, y=y, score=float(row.get("score", 0))))
    return out


def _nearest_gt_dist(gts, x: float, y: float) -> float:
    best = float("inf")
    for gt in gts:
        dx = gt.x - x
        dy = gt.y - y
        best = min(best, dx * dx + dy * dy)
    return best**0.5


def filter_export(
    *,
    in_path: Path,
    out_path: Path,
    verifier_checkpoint: Path,
    trees_parquet: Path,
    verifier_threshold: float = 0.5,
    discovery_m: float = 5.0,
    register_m: float = 3.0,
    radius_m: float = 8.0,
    limit: int | None = None,
    require_chip_rgb: bool = False,
) -> FilterExportResult:
    preds = _load_predictions(in_path)
    if limit is not None:
        preds = preds[:limit]
    if require_chip_rgb:
        from treelidarnet.data.rgb_chip_gate import filter_detections_on_rgb_chips

        preds, rgb_stats = filter_detections_on_rgb_chips(preds, run_id="offline")
        print(
            f"  require-chip-rgb: scoring {rgb_stats['rgb_kept']}/{rgb_stats['input']}",
            flush=True,
        )
    gts = load_ground_truths(trees_parquet, eligible_only=True)

    discovery_in = 0
    discovery_kept = 0
    reg_in = 0
    reg_kept = 0
    for p in preds:
        d = _nearest_gt_dist(gts, p.x, p.y)
        is_disc = d > discovery_m
        is_reg = d <= register_m
        if is_disc:
            discovery_in += 1
        if is_reg:
            reg_in += 1

    scored = score_detections(preds, checkpoint=verifier_checkpoint, radius_m=radius_m)
    kept: list[Detection] = []
    for p in scored:
        vs = p.verifier_score if p.verifier_score is not None else 0.0
        if vs < verifier_threshold:
            continue
        kept.append(p)
        d = _nearest_gt_dist(gts, p.x, p.y)
        if d > discovery_m:
            discovery_kept += 1
        if d <= register_m:
            reg_kept += 1

    with gzip.open(in_path, "rb") as handle:
        payload = json.loads(handle.read())
    pred_rows = detections_to_rows(kept)
    payload["predictions"] = pred_rows
    payload["verifier_checkpoint"] = str(verifier_checkpoint)
    payload["verifier_threshold"] = verifier_threshold
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    result = FilterExportResult(
        input_count=len(preds),
        kept_count=len(kept),
        discovery_input=discovery_in,
        discovery_kept=discovery_kept,
        discovery_retention=discovery_kept / discovery_in if discovery_in else 1.0,
        register_near_input=reg_in,
        register_near_kept=reg_kept,
        register_retention=reg_kept / reg_in if reg_in else 1.0,
    )
    print(
        f"filter-export: {result.kept_count}/{result.input_count} kept; "
        f"discovery retention {result.discovery_kept}/{result.discovery_input} "
        f"({100 * result.discovery_retention:.1f}%); "
        f"register @3m {result.register_near_kept}/{result.register_near_input} "
        f"({100 * result.register_retention:.1f}%)",
        flush=True,
    )
    return result
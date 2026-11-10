"""Evaluate detections vs Gent inventory."""

from __future__ import annotations

import json
from pathlib import Path

import orjson
import pandas as pd

from treelidarnet.metrics import (
    GroundTruth,
    bbox_area_km2,
    discovery_candidates,
    match_detections,
)
from treelidarnet.postprocess import Detection


def load_ground_truths(parquet: Path, *, eligible_only: bool = True) -> list[GroundTruth]:
    df = pd.read_parquet(parquet)
    if eligible_only:
        df = df[df["eligible"]]
    return [GroundTruth(gis_id=row.gis_id, x=row.x, y=row.y) for row in df.itertuples()]


def export_discoveries_geojson(
    preds: list[Detection],
    gts: list[GroundTruth],
    out_path: Path,
    *,
    discovery_distance_m: float = 5.0,
) -> Path:
    discs = discovery_candidates(preds, gts, discovery_distance_m=discovery_distance_m)
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [d.x, d.y]},
            "properties": {"score": d.score, "kind": "discovery"},
        }
        for d in discs
    ]
    payload = {"type": "FeatureCollection", "features": features}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def eval_detections(
    preds: list[Detection],
    gts: list[GroundTruth],
    *,
    run_id: str,
    split: str = "test",
    task: str = "ghent_canopy_detection",
    match_distance_m: float = 3.0,
    extra_meta: dict | None = None,
    export_discoveries: Path | None = None,
    area_km2: float | None = None,
) -> Path:
    result = match_detections(preds, gts, match_distance_m=match_distance_m)
    km2 = area_km2 if area_km2 is not None else bbox_area_km2(preds)
    pred_per_km2 = len(preds) / km2 if km2 > 0 else 0.0
    payload = {
        "version": 2,
        "run_id": run_id,
        "split": split,
        "task": task,
        "label_scope": "stad_gent_managed_inventory_partial",
        "match_distance_m": match_distance_m,
        "pred_count": len(preds),
        "pred_per_km2": pred_per_km2,
        "area_km2": km2,
        "tree_level_audit": {
            "description": "Primary: discoveries (>5m from any GT) excluded from FP",
            "precision": result.precision_audit,
            "recall": result.recall_audit,
            "f1": result.f1_audit,
            "tp": result.tp,
            "fp": result.fp_audit,
            "fn": result.fn,
            "matched_mean_dist_m": result.matched_mean_dist_m,
        },
        "discoveries": {"count": result.discoveries, "threshold_m": 5.0},
        "tree_level_strict": {
            "description": "Secondary: all unmatched preds count as FP (register benchmark)",
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "tp": result.tp,
            "fp": result.fp,
            "fn": result.fn,
            "matched_mean_dist_m": result.matched_mean_dist_m,
        },
    }
    if extra_meta:
        payload.update(extra_meta)
    out = Path("exports/metrics") / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    if export_discoveries is not None:
        export_discoveries_geojson(preds, gts, export_discoveries)
    return out


def eval_from_json_preds(pred_path: Path, trees_parquet: Path, *, run_id: str) -> Path:
    data = json.loads(pred_path.read_text(encoding="utf-8"))
    preds = [Detection(x=p["x"], y=p["y"], score=p.get("score", 1.0)) for p in data["predictions"]]
    gts = load_ground_truths(trees_parquet)
    return eval_detections(preds, gts, run_id=run_id)

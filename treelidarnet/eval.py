"""Evaluate detections vs Gent inventory."""

from __future__ import annotations

import json
from pathlib import Path

import orjson
import pandas as pd

from treelidarnet.metrics import GroundTruth, match_detections
from treelidarnet.postprocess import Detection


def load_ground_truths(parquet: Path, *, eligible_only: bool = True) -> list[GroundTruth]:
    df = pd.read_parquet(parquet)
    if eligible_only:
        df = df[df["eligible"]]
    return [GroundTruth(gis_id=row.gis_id, x=row.x, y=row.y) for row in df.itertuples()]


def eval_detections(
    preds: list[Detection],
    gts: list[GroundTruth],
    *,
    run_id: str,
    split: str = "test",
    match_distance_m: float = 3.0,
    extra_meta: dict | None = None,
) -> Path:
    result = match_detections(preds, gts, match_distance_m=match_distance_m)
    payload = {
        "version": 1,
        "run_id": run_id,
        "split": split,
        "task": "register_tree_detection",
        "label_scope": "stad_gent_managed_inventory_partial",
        "match_distance_m": match_distance_m,
        "tree_level_strict": {
            "description": "All unmatched preds count as FP (benchmark vs register)",
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "tp": result.tp,
            "fp": result.fp,
            "fn": result.fn,
            "matched_mean_dist_m": result.matched_mean_dist_m,
        },
        "tree_level_audit": {
            "description": "Discoveries (>5m from any GT) excluded from FP",
            "precision": result.precision_audit,
            "recall": result.recall_audit,
            "f1": result.f1_audit,
            "tp": result.tp,
            "fp": result.fp_audit,
            "fn": result.fn,
            "matched_mean_dist_m": result.matched_mean_dist_m,
        },
        "discoveries": {"count": result.discoveries, "threshold_m": 5.0},
    }
    if extra_meta:
        payload.update(extra_meta)
    out = Path("exports/metrics") / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return out


def eval_from_json_preds(pred_path: Path, trees_parquet: Path, *, run_id: str) -> Path:
    data = json.loads(pred_path.read_text(encoding="utf-8"))
    preds = [Detection(x=p["x"], y=p["y"], score=p.get("score", 1.0)) for p in data["predictions"]]
    gts = load_ground_truths(trees_parquet)
    return eval_detections(preds, gts, run_id=run_id)

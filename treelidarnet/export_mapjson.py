"""Deck.GL inference bundle (docs/VISUALIZATION.md)."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from treelidarnet.data.crs import lambert_to_wgs84
from treelidarnet.metrics import MatchResult
from treelidarnet.postprocess import Detection


def _tree_radius_m(stamomtrek: float | None) -> tuple[float, bool]:
    if stamomtrek is None:
        return 3.0, False
    r = max(1.5, min(14.0, 0.8 + float(stamomtrek) * 0.04))
    return r, True


def export_inference_gz(
    path: Path,
    *,
    chip_id: str,
    model_id: str,
    gt_rows: list[dict[str, Any]],
    predictions: list[Detection],
    metrics: MatchResult | None = None,
) -> None:
    gt_trees = []
    for row in gt_rows:
        lon, lat = lambert_to_wgs84(row["x"], row["y"])
        radius, has_size = _tree_radius_m(row.get("stamomtrek"))
        gt_trees.append(
            {
                "position": [lon, lat],
                "gis_id": row["gis_id"],
                "slug": row.get("gis_slug", ""),
                "radiusMeters": radius,
                "hasSize": has_size,
                "sortiment": row.get("sortiment", ""),
            }
        )
    preds = []
    for p in predictions:
        lon, lat = lambert_to_wgs84(p.x, p.y)
        preds.append(
            {
                "position": [lon, lat],
                "score": p.score,
                "radiusMeters": 3.0,
            }
        )
    payload: dict[str, Any] = {
        "version": 1,
        "model_id": model_id,
        "inference_at": datetime.now(UTC).isoformat(),
        "chip_id": chip_id,
        "gt_trees": gt_trees,
        "predictions": preds,
    }
    if metrics:
        payload["metrics"] = {
            "precision_at_3m": metrics.precision,
            "recall_at_3m": metrics.recall,
            "f1_at_3m": metrics.f1,
        }
    raw = orjson.dumps(payload, option=orjson.OPT_INDENT_2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(raw)


def export_city_gz(
    path: Path,
    *,
    model_id: str,
    gt_rows: list[dict[str, Any]],
    predictions: list[Detection],
    metrics: MatchResult | None = None,
    chip_count: int = 0,
) -> None:
    gt_trees = []
    for row in gt_rows:
        lon, lat = lambert_to_wgs84(row["x"], row["y"])
        radius, has_size = _tree_radius_m(row.get("stamomtrek"))
        gt_trees.append(
            {
                "position": [lon, lat],
                "gis_id": row["gis_id"],
                "slug": row.get("gis_slug", ""),
                "radiusMeters": radius,
                "hasSize": has_size,
                "sortiment": row.get("sortiment", ""),
            }
        )
    preds = []
    for p in predictions:
        lon, lat = lambert_to_wgs84(p.x, p.y)
        preds.append(
            {
                "position": [lon, lat],
                "score": p.score,
                "radiusMeters": 3.0,
            }
        )
    payload: dict[str, Any] = {
        "version": 2,
        "scope": "ghent_city",
        "model_id": model_id,
        "inference_at": datetime.now(UTC).isoformat(),
        "chip_id": "ghent",
        "chip_count": chip_count,
        "gt_trees": gt_trees,
        "predictions": preds,
    }
    if metrics:
        payload["metrics"] = {
            "precision_at_3m": metrics.precision,
            "recall_at_3m": metrics.recall,
            "f1_at_3m": metrics.f1,
            "precision_audit_at_3m": metrics.precision_audit,
            "f1_audit_at_3m": metrics.f1_audit,
            "discoveries": metrics.discoveries,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
        }
    raw = orjson.dumps(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(raw)
    print(f"gzip {path} ({len(gt_trees)} gt, {len(preds)} pred)", flush=True)

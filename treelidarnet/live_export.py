"""Progress files for live Deck.GL polling (static http.server)."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from treelidarnet.data.crs import lambert_to_wgs84
from treelidarnet.export_mapjson import _tree_radius_m
from treelidarnet.postprocess import Detection


def write_status(path: Path, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **fields,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def export_gt_gz(path: Path, gt_rows: list[dict[str, Any]]) -> None:
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
    raw = orjson.dumps({"gt_trees": gt_trees})
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(raw)


def detections_to_rows(predictions: list[Detection]) -> list[dict[str, Any]]:
    rows = []
    for p in predictions:
        lon, lat = lambert_to_wgs84(p.x, p.y)
        rows.append(
            {
                "position": [lon, lat],
                "score": p.score,
                "radiusMeters": 3.0,
            }
        )
    return rows


def write_live_predictions(path: Path, predictions: list[Detection]) -> None:
    payload = {
        "version": 1,
        "predictions": detections_to_rows(predictions),
    }
    path.write_bytes(orjson.dumps(payload))

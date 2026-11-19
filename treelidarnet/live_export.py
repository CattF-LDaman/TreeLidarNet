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


def detection_row(
    p: Detection,
    *,
    verifier_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    lon, lat = lambert_to_wgs84(p.x, p.y)
    row: dict[str, Any] = {
        "position": [lon, lat],
        "score": p.score,
        "radiusMeters": 3.0,
    }
    if verifier_scores:
        row["verifier_scores"] = verifier_scores
        primary = verifier_scores.get("combo")
        if primary is None:
            primary = verifier_scores.get("v3")
        if primary is None:
            primary = verifier_scores.get("v2")
        if primary is not None:
            row["verifier_score"] = primary
    elif p.verifier_score is not None:
        row["verifier_score"] = p.verifier_score
    return row


def detections_to_rows(
    predictions: list[Detection],
    *,
    verifier_scores_list: list[dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for i, p in enumerate(predictions):
        vs = verifier_scores_list[i] if verifier_scores_list and i < len(verifier_scores_list) else None
        rows.append(detection_row(p, verifier_scores=vs))
    return rows


def read_live_status(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return orjson.loads(path.read_bytes())


def read_live_predictions(path: Path) -> list[Detection]:
    """Load Lambert detections from a live predictions.json written by infer-gent."""
    from treelidarnet.data.crs import wgs84_to_lambert

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = orjson.loads(path.read_bytes())
    rows = payload.get("predictions") or []
    out: list[Detection] = []
    for row in rows:
        lon, lat = float(row["position"][0]), float(row["position"][1])
        x, y = wgs84_to_lambert(lon, lat)
        vs = row.get("verifier_score")
        out.append(
            Detection(
                x=x,
                y=y,
                score=float(row["score"]),
                verifier_score=float(vs) if vs is not None else None,
            )
        )
    return out


def write_live_predictions(
    path: Path,
    predictions: list[Detection],
    *,
    verifier_scores_list: list[dict[str, float]] | None = None,
) -> None:
    payload = {
        "version": 1,
        "compare": verifier_scores_list is not None,
        "predictions": detections_to_rows(
            predictions, verifier_scores_list=verifier_scores_list
        ),
    }
    path.write_bytes(orjson.dumps(payload))

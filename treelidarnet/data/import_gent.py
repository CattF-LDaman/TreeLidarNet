"""Import Stad Gent trees → parquet."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from treelidarnet.data.crs import wgs84_to_lambert
from treelidarnet.data.eligibility import eligibility
from treelidarnet.data.identifiers import gis_id_to_slug

DEFAULT_EXPORT_URL = (
    "https://data.stad.gent/api/explore/v2.1/catalog/datasets/locaties-bomen-gent/exports/geojson"
)


def _fetch_geojson(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def _parse_feature(feature: dict[str, Any]) -> dict[str, Any] | None:
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "Point":
        return None
    coords = geometry.get("coordinates") or []
    if len(coords) < 2:
        return None
    props = feature.get("properties") or {}
    gis_id = props.get("gis_id")
    if not gis_id:
        return None
    lon, lat = float(coords[0]), float(coords[1])
    st = props.get("stamomtrek")
    try:
        st_val = float(st) if st is not None else None
    except (TypeError, ValueError):
        st_val = None
    aanleg = props.get("aanlegjaar")
    aanlegjaar = "" if aanleg is None else str(aanleg).strip()
    sortiment = props.get("sortiment") or ""
    beheerfase = props.get("beheerfase") or ""
    eligible, reason = eligibility(
        aanlegjaar=aanlegjaar,
        beheerfase=beheerfase,
        sortiment=sortiment,
    )
    x, y = wgs84_to_lambert(lon, lat)
    return {
        "gis_id": str(gis_id),
        "gis_slug": gis_id_to_slug(str(gis_id)),
        "sortiment": sortiment,
        "straatnaam": props.get("straatnaam") or "",
        "stamomtrek": st_val,
        "eigenaar": props.get("eigenaar") or "",
        "beheerfase": beheerfase,
        "aanlegjaar": aanlegjaar,
        "plantwijze": str(props.get("plantwijze") or "").strip(),
        "lon": lon,
        "lat": lat,
        "x": x,
        "y": y,
        "eligible": eligible,
        "ineligible_reason": reason,
    }


def import_gent_geojson(
    *,
    url: str = DEFAULT_EXPORT_URL,
    out: Path,
) -> dict[str, Any]:
    payload = _fetch_geojson(url)
    features = payload.get("features") or []
    rows = []
    for feat in features:
        parsed = _parse_feature(feat)
        if parsed:
            rows.append(parsed)
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    raw = out.read_bytes()
    return {
        "url": url,
        "fetched_at": datetime.now(UTC).isoformat(),
        "record_count": len(df),
        "eligible_count": int(df["eligible"].sum()) if len(df) else 0,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "path": str(out),
    }

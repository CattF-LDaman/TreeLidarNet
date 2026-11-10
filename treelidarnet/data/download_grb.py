"""Download GRB vectors for Gent bbox (OGC API Features)."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from treelidarnet.paths import data_dir, manifests_dir

OGC_BASE = "https://geo.api.vlaanderen.be/GRB/ogc/features/v1"
BUILDING_COLLECTION = "IngeschetstGebouw"
ROAD_COLLECTION = "Wegsegment"
BBOX_CRS = "http://www.opengis.net/def/crs/EPSG/0/31370"
PAGE_LIMIT = 1000


def bbox_lambert_from_trees(
    trees_parquet: Path,
    *,
    buffer_m: float = 500.0,
) -> tuple[float, float, float, float]:
    df = pd.read_parquet(trees_parquet)
    xmin = float(df["x"].min()) - buffer_m
    xmax = float(df["x"].max()) + buffer_m
    ymin = float(df["y"].min()) - buffer_m
    ymax = float(df["y"].max()) + buffer_m
    return xmin, ymin, xmax, ymax


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "TreeLidarNet/0.1"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _items_url(collection: str, bbox: tuple[float, float, float, float], *, start_index: int) -> str:
    b = ",".join(str(v) for v in bbox)
    q = urllib.parse.urlencode(
        {
            "f": "json",
            "bbox": b,
            "bbox-crs": BBOX_CRS,
            "limit": PAGE_LIMIT,
            "startIndex": start_index,
        }
    )
    return f"{OGC_BASE}/collections/{collection}/items?{q}"


def fetch_collection_geojson(
    collection: str,
    bbox: tuple[float, float, float, float],
) -> dict:
    features: list[dict] = []
    seen: set[str] = set()
    start_index = 0
    while True:
        url = _items_url(collection, bbox, start_index=start_index)
        for attempt in range(5):
            try:
                payload = _fetch_json(url)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        batch = payload.get("features") or []
        for feat in batch:
            fid = str(feat.get("id", ""))
            if fid and fid in seen:
                continue
            if fid:
                seen.add(fid)
            features.append(feat)
        if not batch or len(batch) < PAGE_LIMIT:
            break
        start_index += len(batch)
        if start_index % 10_000 == 0:
            print(f"  {collection} startIndex {start_index} features={len(features)}", flush=True)
    return {"type": "FeatureCollection", "features": features}


def download_gent_grb(
    *,
    trees_parquet: Path,
    dest_dir: Path | None = None,
    buffer_m: float = 500.0,
) -> dict:
    dest_dir = dest_dir or (data_dir() / "raw" / "grb")
    dest_dir.mkdir(parents=True, exist_ok=True)
    bbox = bbox_lambert_from_trees(trees_parquet, buffer_m=buffer_m)
    print(f"GRB bbox (EPSG:31370): {bbox}", flush=True)

    buildings_path = dest_dir / "gent_grb_buildings.geojson"
    roads_path = dest_dir / "gent_grb_roads.geojson"

    print(f"Fetching {BUILDING_COLLECTION}...", flush=True)
    b_geo = fetch_collection_geojson(BUILDING_COLLECTION, bbox)
    buildings_path.write_text(json.dumps(b_geo), encoding="utf-8")
    print(f"  wrote {len(b_geo['features'])} buildings → {buildings_path}", flush=True)

    print(f"Fetching {ROAD_COLLECTION}...", flush=True)
    r_geo = fetch_collection_geojson(ROAD_COLLECTION, bbox)
    roads_path.write_text(json.dumps(r_geo), encoding="utf-8")
    print(f"  wrote {len(r_geo['features'])} roads → {roads_path}", flush=True)

    meta = {
        "bbox": bbox,
        "buffer_m": buffer_m,
        "building_count": len(b_geo["features"]),
        "road_count": len(r_geo["features"]),
        "downloaded_at": datetime.now(tz=UTC).isoformat(),
        "collections": {
            "buildings": BUILDING_COLLECTION,
            "roads": ROAD_COLLECTION,
        },
    }
    meta_path = manifests_dir() / "grb_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta

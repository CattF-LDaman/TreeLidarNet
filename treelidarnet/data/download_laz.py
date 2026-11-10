"""Download DHMV II LAZ tiles via OpenLiDAR WMS GetFeatureInfo."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from treelidarnet.paths import manifests_dir

WMS = "https://remotesensing.vlaanderen.be/services/openlidar/wms"
DOWNLOAD_BASE = "https://remotesensing.vlaanderen.be/download/openlidar/"
LAYER = "LiDAR_DHMV_II_LAZtiles"
_LAZ_PATH_RE = re.compile(r"(LiDAR_DHMV_2[^\s<\"]+\.laz)", re.I)


def tile_corners_for_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float, *, step: int = 500
) -> list[tuple[int, int]]:
    x0 = int(xmin // step) * step
    y0 = int(ymin // step) * step
    x1 = int(xmax // step) * step
    y1 = int(ymax // step) * step
    return [(x, y) for x in range(x0, x1 + 1, step) for y in range(y0, y1 + 1, step)]


def _getfeatureinfo(tile_x: int, tile_y: int) -> str:
    bbox = f"{tile_x},{tile_y},{tile_x + 500},{tile_y + 500}"
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": LAYER,
        "QUERY_LAYERS": LAYER,
        "CRS": "EPSG:31370",
        "BBOX": bbox,
        "WIDTH": "256",
        "HEIGHT": "256",
        "I": "128",
        "J": "128",
        "INFO_FORMAT": "text/html",
    }
    url = WMS + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "TreeLidarNet/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8", errors="replace")


def laz_paths_from_html(html: str) -> list[str]:
    paths = [m.group(1).strip() for m in _LAZ_PATH_RE.finditer(html)]
    return list(dict.fromkeys(paths))


def download_laz_file(rel_path: str, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(rel_path).name
    dest = dest_dir / name
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    url = DOWNLOAD_BASE + rel_path.lstrip("/")
    req = urllib.request.Request(url, headers={"User-Agent": "TreeLidarNet/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return dest
    except Exception:
        return None


def download_tile(tile_x: int, tile_y: int, dest_dir: Path) -> list[Path]:
    html = _getfeatureinfo(tile_x, tile_y)
    paths = laz_paths_from_html(html)
    saved: list[Path] = []
    for rel in paths:
        p = download_laz_file(rel, dest_dir)
        if p:
            saved.append(p)
    return saved


def gent_bbox_from_parquet(parquet: Path) -> tuple[float, float, float, float]:
    df = pd.read_parquet(parquet)
    if "eligible" in df.columns:
        df = df[df["eligible"]]
    return float(df.x.min()), float(df.y.min()), float(df.x.max()), float(df.y.max())


def download_gent_laz(
    *,
    trees_parquet: Path,
    dest_dir: Path,
    max_workers: int = 6,
    limit: int | None = None,
) -> Path:
    xmin, ymin, xmax, ymax = gent_bbox_from_parquet(trees_parquet)
    corners = tile_corners_for_bbox(xmin, ymin, xmax, ymax)
    if limit:
        corners = corners[:limit]
    manifest_rows = []
    dest_dir.mkdir(parents=True, exist_ok=True)

    def work(corner: tuple[int, int]) -> tuple[int, int, list[str]]:
        tx, ty = corner
        try:
            html = _getfeatureinfo(tx, ty)
            paths = laz_paths_from_html(html)
            names = []
            for rel in paths:
                p = download_laz_file(rel, dest_dir)
                if p:
                    names.append(p.name)
            return tx, ty, names
        except Exception:
            return tx, ty, []

    manifest = manifests_dir() / "gent_laz_manifest.json"

    def flush_manifest() -> None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(manifest_rows, indent=2) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(work, c) for c in corners]
        for i, fut in enumerate(as_completed(futs), 1):
            tx, ty, names = fut.result()
            manifest_rows.append({"tile_x": tx, "tile_y": ty, "files": names})
            if i % 25 == 0 or i == len(corners):
                flush_manifest()
                print(f"LAZ tiles {i}/{len(corners)}", flush=True)

    flush_manifest()
    return manifest

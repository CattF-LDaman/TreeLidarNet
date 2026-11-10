"""EPSG:4326 ↔ EPSG:31370 (Lambert 72)."""

from __future__ import annotations

from pyproj import Transformer

_WGS84_TO_LAMBERT = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)
_LAMBERT_TO_WGS84 = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)


def wgs84_to_lambert(lon: float, lat: float) -> tuple[float, float]:
    x, y = _WGS84_TO_LAMBERT.transform(lon, lat)
    return float(x), float(y)


def lambert_to_wgs84(x: float, y: float) -> tuple[float, float]:
    lon, lat = _LAMBERT_TO_WGS84.transform(x, y)
    return float(lon), float(lat)

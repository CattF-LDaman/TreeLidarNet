"""GIS id ↔ URL slug (HugATree-compatible)."""

from __future__ import annotations

from urllib.parse import quote, unquote


def gis_id_to_slug(gis_id: str) -> str:
    return quote(gis_id.replace("/", "-"), safe="-_.")


def slug_to_gis_id(slug: str) -> str:
    return unquote(slug).replace("-", "/")

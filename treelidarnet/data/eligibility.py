"""Label eligibility rules (docs/DATA.md)."""

from __future__ import annotations

import re

LIDAR_YEAR_MAX = 2015
EXCLUDE_BEHEERFASE = ("Afbraak", "Verwijderd", "Sloop")


def parse_aanlegjaar(raw: str) -> int | None:
    s = (raw or "").strip()
    if not s:
        return None
    m = re.match(r"^(\d{4})", s)
    if not m:
        return None
    return int(m.group(1))


def eligibility(
    *,
    aanlegjaar: str,
    beheerfase: str,
    sortiment: str,
    lidar_year_max: int = LIDAR_YEAR_MAX,
    exclude_beheerfase: tuple[str, ...] = EXCLUDE_BEHEERFASE,
) -> tuple[bool, str | None]:
    _ = sortiment
    year = parse_aanlegjaar(aanlegjaar)
    if year is not None and year > lidar_year_max:
        return False, "aanlegjaar_after_lidar"
    phase = (beheerfase or "").strip()
    for pat in exclude_beheerfase:
        if pat.lower() in phase.lower():
            return False, f"beheerfase:{pat}"
    return True, None

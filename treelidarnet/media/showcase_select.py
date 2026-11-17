"""Pick diverse high-confidence verifier showcase detections on RGB-patched chips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from treelidarnet.data.crops import chip_id_for_point
from treelidarnet.data.crs import wgs84_to_lambert
from treelidarnet.paths import chips_dir


@dataclass(frozen=True)
class ShowcasePick:
    lon: float
    lat: float
    x: float
    y: float
    bev_score: float
    verifier_score: float
    chip_id: str

    @property
    def quality(self) -> float:
        return self.bev_score * self.verifier_score


def load_verified_predictions(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        import gzip

        raw = gzip.decompress(raw)
    data = json.loads(raw)
    return list(data.get("predictions") or [])


def chip_has_rgb(chip_id: str, *, chip_root: Path | None = None) -> bool:
    root = chip_root or chips_dir()
    npz = root / f"{chip_id}.npz"
    if not npz.is_file():
        return False
    data = np.load(npz)
    try:
        if "rgb" not in data.files:
            return False
        return len(data["rgb"]) == len(data["xyz"])
    finally:
        data.close()


def _candidates_from_rows(
    rows: list[dict],
    *,
    bev_min: float,
    verifier_min: float,
    chip_root: Path | None,
) -> list[ShowcasePick]:
    out: list[ShowcasePick] = []
    for row in rows:
        bev = float(row.get("score", 0.0))
        vs = row.get("verifier_score")
        if vs is None:
            continue
        vs_f = float(vs)
        if bev < bev_min or vs_f < verifier_min:
            continue
        lon, lat = float(row["position"][0]), float(row["position"][1])
        x, y = wgs84_to_lambert(lon, lat)
        cid = chip_id_for_point(x, y)
        if not chip_has_rgb(cid, chip_root=chip_root):
            continue
        out.append(
            ShowcasePick(
                lon=lon,
                lat=lat,
                x=x,
                y=y,
                bev_score=bev,
                verifier_score=vs_f,
                chip_id=cid,
            )
        )
    out.sort(key=lambda p: p.quality, reverse=True)
    return out


def farthest_point_select(
    candidates: list[ShowcasePick],
    *,
    n: int,
    min_spacing_m: float,
) -> list[ShowcasePick]:
    """Greedy: start from best quality, add points at least min_spacing_m from picks."""
    if not candidates or n <= 0:
        return []
    chosen: list[ShowcasePick] = [candidates[0]]
    for cand in candidates[1:]:
        if len(chosen) >= n:
            break
        ok = True
        for picked in chosen:
            dx = cand.x - picked.x
            dy = cand.y - picked.y
            if dx * dx + dy * dy < min_spacing_m * min_spacing_m:
                ok = False
                break
        if ok:
            chosen.append(cand)
    if len(chosen) < n:
        for cand in candidates:
            if cand in chosen:
                continue
            chosen.append(cand)
            if len(chosen) >= n:
                break
    return chosen[:n]


def select_showcase_picks(
    verified_path: Path,
    *,
    n: int = 20,
    bev_min: float = 0.85,
    verifier_min: float = 0.85,
    min_spacing_m: float = 400.0,
    chip_root: Path | None = None,
) -> list[ShowcasePick]:
    rows = load_verified_predictions(verified_path)
    candidates = _candidates_from_rows(
        rows,
        bev_min=bev_min,
        verifier_min=verifier_min,
        chip_root=chip_root,
    )
    return farthest_point_select(candidates, n=n, min_spacing_m=min_spacing_m)
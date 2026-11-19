"""Client/server shared verifier combine rules for explorer filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CombineConfig:
    mode: str = "either_or"
    bev_min: float = 0.85
    t_v2: float = 0.5
    t_v3: float = 0.5
    t_combo: float = 0.5
    w_v2: float = 0.5
    w_v3: float = 0.5
    register_split_m: float = 5.0
    disagree_delta: float = 0.15
    discoveries_only: bool = False
    discovery_min_m: float = 2.0
    require_rgb: bool = False


def detection_passes(
    *,
    bev: float,
    v2: float,
    v3: float,
    dist_register_m: float,
    has_rgb: bool,
    cfg: CombineConfig,
) -> bool:
    if bev < cfg.bev_min:
        return False
    if cfg.require_rgb and not has_rgb:
        return False
    if cfg.discoveries_only and dist_register_m <= cfg.discovery_min_m:
        return False

    mode = (cfg.mode or "either_or").lower()
    if mode == "disagreement":
        return abs(v2 - v3) >= cfg.disagree_delta

    if mode == "v2_only":
        return v2 >= cfg.t_v2
    if mode == "v3_only":
        return v3 >= cfg.t_v3
    if mode == "both_and":
        return v2 >= cfg.t_v2 and v3 >= cfg.t_v3
    if mode == "either_or":
        return v2 >= cfg.t_v2 or v3 >= cfg.t_v3
    if mode == "max_score":
        return max(v2, v3) >= cfg.t_combo
    if mode == "min_score":
        return min(v2, v3) >= cfg.t_combo
    if mode == "weighted":
        return cfg.w_v2 * v2 + cfg.w_v3 * v3 >= cfg.t_combo
    if mode == "register_split":
        if dist_register_m <= cfg.register_split_m:
            return v3 >= cfg.t_v3
        return v2 >= cfg.t_v2
    return v2 >= cfg.t_v2 or v3 >= cfg.t_v3


def combine_config_from_dict(data: dict[str, Any]) -> CombineConfig:
    known = {f.name for f in CombineConfig.__dataclass_fields__.values()}
    return CombineConfig(**{k: v for k, v in data.items() if k in known})


def weighted_combo(v2: float, v3: float, *, w_v2: float = 0.5, w_v3: float = 0.5) -> float:
    return w_v2 * v2 + w_v3 * v3


def export_keep_passes(
    *,
    v2: float,
    v3: float,
    combine_mode: str,
    threshold: float,
    w_v2: float = 0.5,
    w_v3: float = 0.5,
    t_v2: float | None = None,
    t_v3: float | None = None,
) -> bool:
    """Verifier export keep rule on dual scores (BEV pre-filter already applied)."""
    t2 = threshold if t_v2 is None else t_v2
    t3 = threshold if t_v3 is None else t_v3
    mode = (combine_mode or "weighted").lower()
    if mode == "v2_only":
        return v2 >= t2
    if mode == "v3_only":
        return v3 >= t3
    if mode == "both_and":
        return v2 >= t2 and v3 >= t3
    if mode == "either_or":
        return v2 >= t2 or v3 >= t3
    if mode == "max_score":
        return max(v2, v3) >= threshold
    if mode == "min_score":
        return min(v2, v3) >= threshold
    if mode == "weighted":
        return weighted_combo(v2, v3, w_v2=w_v2, w_v3=w_v3) >= threshold
    return weighted_combo(v2, v3, w_v2=w_v2, w_v3=w_v3) >= threshold


def export_primary_score(
    v2: float,
    v3: float,
    *,
    combine_mode: str,
    w_v2: float = 0.5,
    w_v3: float = 0.5,
) -> float:
    """Single scalar stored as verifier_score on export rows."""
    mode = (combine_mode or "weighted").lower()
    if mode == "weighted":
        return weighted_combo(v2, v3, w_v2=w_v2, w_v3=w_v3)
    if mode == "v2_only":
        return v2
    if mode == "v3_only":
        return v3
    if mode == "max_score":
        return max(v2, v3)
    if mode == "min_score":
        return min(v2, v3)
    return weighted_combo(v2, v3, w_v2=w_v2, w_v3=w_v3)
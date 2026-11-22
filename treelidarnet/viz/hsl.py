"""HSL colors aligned with explorer/explorer_style.js."""

from __future__ import annotations

import numpy as np

def clamp01(v: float | np.ndarray) -> float | np.ndarray:
    return np.clip(v, 0.0, 1.0)


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    """Single-pixel HSL to RGB (0–255), matching Deck explorer Style."""
    h = float(h) % 360.0
    s = float(clamp01(s))
    light = float(clamp01(l))
    c = (1.0 - abs(2.0 * light - 1.0)) * s
    x = c * (1.0 - abs(((h / 60.0) % 2.0) - 1.0))
    m = light - c / 2.0
    if h < 60:
        r, g, b = c, x, 0.0
    elif h < 120:
        r, g, b = x, c, 0.0
    elif h < 180:
        r, g, b = 0.0, c, x
    elif h < 240:
        r, g, b = 0.0, x, c
    elif h < 300:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return (
        int(round((r + m) * 255)),
        int(round((g + m) * 255)),
        int(round((b + m) * 255)),
    )


def combo_colors(
    v2: np.ndarray,
    v3: np.ndarray,
    *,
    w_v2: float = 0.5,
    w_v3: float = 0.5,
    hue_low: float = 0.0,
    hue_high: float = 142.0,
    sat: float = 0.38,
    light: float = 0.52,
) -> np.ndarray:
    """RGB uint8 (N, 3) from raw v2/v3 using fixed hue ramp on combo score."""
    v2 = np.asarray(v2, dtype=np.float64)
    v3 = np.asarray(v3, dtype=np.float64)
    n = len(v2)
    # same formula as treelidarnet.explorer.combine.weighted_combo
    combo = w_v2 * v2 + w_v3 * v3
    t = clamp01(combo)
    hues = hue_low + (hue_high - hue_low) * t
    out = np.empty((n, 3), dtype=np.uint8)
    for i in range(n):
        out[i] = hsl_to_rgb(float(hues[i]), sat, light)
    return out


def combo_from_arrays(
    v2: np.ndarray,
    v3: np.ndarray,
    *,
    w_v2: float = 0.5,
    w_v3: float = 0.5,
) -> np.ndarray:
    v2 = np.asarray(v2, dtype=np.float64)
    v3 = np.asarray(v3, dtype=np.float64)
    return w_v2 * v2 + w_v3 * v3
"""Raster GIF: weighted combo color + bouncing combo threshold."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from treelidarnet.data.crs import wgs84_to_lambert
from treelidarnet.viz.hsl import combo_colors, combo_from_arrays

BG_RGB = (15, 20, 25)


def bounce_thresholds(low: float, high: float, step: float) -> list[float]:
    up = list(np.arange(low, high + step * 0.5, step))
    down = list(np.arange(high - step, low - step * 0.5, -step))
    return up + down


def load_verified_export(export_gz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    with gzip.open(export_gz, "rt", encoding="utf-8") as f:
        data = json.load(f)
    xs, ys, v2s, v3s = [], [], [], []
    for p in data.get("predictions") or []:
        lon, lat = p["position"]
        x, y = wgs84_to_lambert(lon, lat)
        vs = p.get("verifier_scores") or {}
        if vs:
            v2 = float(vs.get("v2", 0.0))
            v3 = float(vs.get("v3", 0.0))
        else:
            s = float(p.get("verifier_score") or p.get("score") or 0.0)
            v2 = v3 = s
        xs.append(x)
        ys.append(y)
        v2s.append(v2)
        v3s.append(v3)
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(v2s, dtype=np.float64),
        np.asarray(v3s, dtype=np.float64),
        data,
    )


def _lambert_to_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    rx = xmax - xmin
    ry = ymax - ymin
    if rx <= 0:
        rx = 1.0
    if ry <= 0:
        ry = 1.0
    px = np.clip(((xs - xmin) / rx * (width - 1)).astype(np.int32), 0, width - 1)
    py = np.clip(((ymax - ys) / ry * (height - 1)).astype(np.int32), 0, height - 1)
    return px, py


def _bbox_for_canvas(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    pad: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    xmin, xmax = float(xs.min() - pad), float(xs.max() + pad)
    ymin, ymax = float(ys.min() - pad), float(ys.max() + pad)
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    rx = xmax - xmin
    ry = ymax - ymin
    target = float(width) / float(height)
    data = rx / ry if ry > 0 else target
    if data > target:
        ry = rx / target
    else:
        rx = ry * target
    return cx - rx * 0.5, cx + rx * 0.5, cy - ry * 0.5, cy + ry * 0.5


def _bbox_center_zoom(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    pad: float,
    width: int,
    height: int,
    zoom: float,
    center_x: float | None = None,
    center_y: float | None = None,
) -> tuple[float, float, float, float]:
    """Viewport centered on the urban core; zoom>1 narrows Lambert span (more pixels per tree)."""
    if zoom <= 0:
        raise ValueError("zoom must be positive")
    cx = float(np.median(xs) if center_x is None else center_x)
    cy = float(np.median(ys) if center_y is None else center_y)
    xmin, xmax = float(xs.min() - pad), float(xs.max() + pad)
    ymin, ymax = float(ys.min() - pad), float(ys.max() + pad)
    rx = (xmax - xmin) / zoom
    ry = (ymax - ymin) / zoom
    target = float(width) / float(height)
    data = rx / ry if ry > 0 else target
    if data > target:
        ry = rx / target
    else:
        rx = ry * target
    return cx - rx * 0.5, cx + rx * 0.5, cy - ry * 0.5, cy + ry * 0.5


def _splat_rgb(
    canvas: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    point_radius: int,
) -> None:
    if not mask.any():
        return
    pxm = px[mask]
    pym = py[mask]
    rgbm = rgb[mask]
    canvas[pym, pxm] = rgbm
    if point_radius < 1:
        return
    offsets = [(dx, dy) for dx in range(-point_radius, point_radius + 1) for dy in range(-point_radius, point_radius + 1)]
    for dx, dy in offsets:
        if dx == 0 and dy == 0:
            continue
        nx = np.clip(pxm + dx, 0, width - 1)
        ny = np.clip(pym + dy, 0, height - 1)
        canvas[ny, nx] = rgbm


def raster_frame(
    px: np.ndarray,
    py: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    title_h: int,
) -> Image.Image:
    canvas = np.full((height, width, 3), BG_RGB, dtype=np.uint8)
    _splat_rgb(canvas, px, py, rgb, mask, width=width, height=height, point_radius=0)
    body = Image.fromarray(canvas, mode="RGB")
    if title_h <= 0:
        return body
    out_h = height + title_h
    out = Image.new("RGB", (width, out_h), BG_RGB)
    out.paste(body, (0, title_h))
    return out


def render_combo_export_static(
    *,
    export_gz: Path,
    out_path: Path,
    threshold: float = 0.45,
    width: int = 3840,
    height: int = 2160,
    pad: float = 400.0,
    zoom: float = 1.0,
    point_radius: int = 0,
    hue_low: float = 0.0,
    hue_high: float = 142.0,
    sat: float = 0.38,
    light: float = 0.52,
    w_v2: float = 0.5,
    w_v3: float = 0.5,
) -> None:
    xs, ys, v2, v3, _meta = load_verified_export(export_gz)
    n = len(xs)
    if n == 0:
        raise ValueError(f"no predictions in {export_gz}")

    combo = combo_from_arrays(v2, v3, w_v2=w_v2, w_v3=w_v3)
    rgb = combo_colors(
        v2, v3, w_v2=w_v2, w_v3=w_v3, hue_low=hue_low, hue_high=hue_high, sat=sat, light=light
    )
    if zoom > 1.0:
        xmin, xmax, ymin, ymax = _bbox_center_zoom(
            xs, ys, pad=pad, width=width, height=height, zoom=zoom
        )
    else:
        xmin, xmax, ymin, ymax = _bbox_for_canvas(xs, ys, pad=pad, width=width, height=height)
    px, py = _lambert_to_pixels(
        xs, ys, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width=width, height=height
    )
    score_mask = combo >= threshold
    in_view = (
        (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax) & score_mask
    )
    canvas = np.full((height, width, 3), BG_RGB, dtype=np.uint8)
    _splat_rgb(
        canvas, px, py, rgb, in_view, width=width, height=height, point_radius=point_radius
    )
    img = Image.fromarray(canvas, mode="RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)
    span_km = (xmax - xmin) / 1000.0
    print(
        f"Wrote {out_path} ({width}×{height}, zoom={zoom:.2f}, ~{span_km:.1f} km wide, "
        f"combo≥{threshold:.2f}, {int(in_view.sum()):,} in view / {n:,} total)",
        flush=True,
    )


def render_combo_export_gif(
    *,
    export_gz: Path,
    out_path: Path,
    low: float = 0.45,
    high: float = 1.0,
    step: float = 0.02,
    duration_ms: int = 160,
    canvas_px: int = 2400,
    title_h: int = 72,
    hue_low: float = 0.0,
    hue_high: float = 142.0,
    sat: float = 0.38,
    light: float = 0.52,
    w_v2: float = 0.5,
    w_v3: float = 0.5,
) -> int:
    xs, ys, v2, v3, meta = load_verified_export(export_gz)
    n = len(xs)
    if n == 0:
        raise ValueError(f"no predictions in {export_gz}")

    combo = combo_from_arrays(v2, v3, w_v2=w_v2, w_v3=w_v3)
    rgb = combo_colors(
        v2, v3, w_v2=w_v2, w_v3=w_v3, hue_low=hue_low, hue_high=hue_high, sat=sat, light=light
    )

    pad = 400.0
    xmin, xmax = float(xs.min() - pad), float(xs.max() + pad)
    ymin, ymax = float(ys.min() - pad), float(ys.max() + pad)
    px, py = _lambert_to_pixels(
        xs, ys, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width=canvas_px, height=canvas_px
    )

    thresholds = bounce_thresholds(low, high, step)
    frames: list[Image.Image] = []
    w_label = meta.get("verifier_combine_mode") or meta.get("verifier_threshold")
    subtitle = (
        f"weighted {w_v2:.1f}·v2 + {w_v3:.1f}·v3 · hue {hue_low:.0f}°–{hue_high:.0f}° sat={sat:.2f}"
    )

    for thr in thresholds:
        mask = combo >= thr
        img = raster_frame(
            px, py, rgb, mask, width=canvas_px, height=canvas_px, title_h=title_h
        )
        draw = ImageDraw.Draw(img)
        title = f"Ghent verified export — min combo {thr:.2f}  ·  {int(mask.sum()):,} / {n:,} shown"
        draw.text((12, 8), title, fill=(231, 233, 234))
        draw.text((12, 34), subtitle, fill=(140, 150, 160))
        frames.append(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(
        f"Wrote {out_path} ({len(frames)} frames, {duration_ms} ms/frame, {n:,} detections)",
        flush=True,
    )
    return len(frames)
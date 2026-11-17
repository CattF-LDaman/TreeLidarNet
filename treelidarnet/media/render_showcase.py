"""Render map-zoom and RGB splat frames for verifier showcase reels."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

from treelidarnet.data.crops import IN_CHANNELS_RGB, extract_crop_at_global
from treelidarnet.media.showcase_select import ShowcasePick

if TYPE_CHECKING:
    from pathlib import Path


def _ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def render_map_zoom_frames(
    pick: ShowcasePick,
    *,
    register_xy: np.ndarray | None,
    city_bounds: tuple[float, float, float, float],
    n_frames: int = 27,
    figsize: tuple[float, float] = (8, 8),
    wide_half_m: float = 2500.0,
    tight_half_m: float = 90.0,
    dpi: int = 96,
    on_frame: callable | None = None,
) -> list[Image.Image]:
    """Lambert map zoom toward one detection."""
    cx, cy = pick.x, pick.y
    frames: list[Image.Image] = []
    for i in range(n_frames):
        t = _ease_in_out(i / max(n_frames - 1, 1))
        half = wide_half_m + (tight_half_m - wide_half_m) * t
        xmin, xmax = cx - half, cx + half
        ymin, ymax = cy - half, cy + half

        fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1419", dpi=dpi)
        ax.set_facecolor("#0f1419")
        if register_xy is not None and len(register_xy):
            m = (
                (register_xy[:, 0] >= xmin)
                & (register_xy[:, 0] <= xmax)
                & (register_xy[:, 1] >= ymin)
                & (register_xy[:, 1] <= ymax)
            )
            sub = register_xy[m]
            if len(sub):
                ax.scatter(
                    sub[:, 0],
                    sub[:, 1],
                    s=3,
                    c="#34d399",
                    alpha=0.35,
                    linewidths=0,
                    rasterized=True,
                )
        ring = Circle(
            (cx, cy),
            radius=3.0,
            fill=False,
            edgecolor="#fbbf24",
            linewidth=2.5,
            alpha=0.95,
        )
        ax.add_patch(ring)
        ax.scatter([cx], [cy], s=40, c="#fbbf24", edgecolors="#fff", linewidths=0.6, zorder=5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            f"BEV {pick.bev_score:.2f} · Verifier {pick.verifier_score:.2f}",
            transform=ax.transAxes,
            color="#e7e9ea",
            fontsize=11,
            va="top",
        )
        ax.text(
            0.02,
            0.02,
            "Ghent · DHMV II LiDAR",
            transform=ax.transAxes,
            color="#8b949e",
            fontsize=9,
            va="bottom",
        )
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf))
        plt.close(fig)
        if on_frame is not None:
            on_frame(i + 1, n_frames, "map")
    return frames


# Downward isometric orbit: classic cone angle from +Z (≈54.7°) oblique top-down.
_ISO_ZENITH_RAD = math.atan(math.sqrt(2.0))
_BG_RGB = np.array([0.055, 0.078, 0.096], dtype=np.float32)
_SPLAT_ALPHA_THRESHOLD = 0.06
# Screen-space Gaussian radius (px) — larger reads more “splatty” on 960² frames.
_SPLAT_SIGMA_MIN = 2.6
_SPLAT_SIGMA_DEPTH = 1.85

_KERNEL_CACHE: dict[tuple[str, float], np.ndarray] = {}


def height_colors(z: np.ndarray) -> np.ndarray:
    """Match viewer/crop_viewer_3d.js heightColor (emerald → warm crown)."""
    z_min = float(z.min())
    z_max = float(z.max())
    if z_max <= z_min:
        t = np.full(len(z), 0.5, dtype=np.float64)
    else:
        t = (z.astype(np.float64) - z_min) / (z_max - z_min)
    r = (52.0 + t * 200.0) / 255.0
    g = (211.0 - t * 80.0) / 255.0
    b = (153.0 + t * 40.0) / 255.0
    return np.stack([r, g, b], axis=1).astype(np.float32)


def _kernel(kind: str, radius_or_sigma: float) -> np.ndarray:
    key = (kind, round(radius_or_sigma, 3))
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    if kind == "disk":
        r_px = max(1, int(math.ceil(radius_or_sigma)))
        size = 2 * r_px + 1
        cy = cx = r_px
        yy, xx = np.ogrid[-cy : size - cy, -cx : size - cx]
        k = (xx * xx + yy * yy <= radius_or_sigma * radius_or_sigma).astype(np.float32)
    else:
        sig = max(radius_or_sigma, 0.35)
        r_px = max(2, int(math.ceil(sig * 2.8)))
        size = 2 * r_px + 1
        cy = cx = r_px
        yy, xx = np.ogrid[-cy : size - cy, -cx : size - cx]
        k = np.exp(-(xx * xx + yy * yy) / (2.0 * sig * sig)).astype(np.float32)
    _KERNEL_CACHE[key] = k
    return k


def _look_at_downward_orbit(
    yaw: float,
    *,
    extent_m: float,
    zenith_rad: float = _ISO_ZENITH_RAD,
    distance_scale: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Camera above the XY plane on a ring, looking at the crop center (Z-up)."""
    r = max(extent_m * distance_scale, 6.0)
    st, ct = math.sin(zenith_rad), math.cos(zenith_rad)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cam_pos = np.array([r * st * cy, r * st * sy, r * ct], dtype=np.float64)
    target = np.zeros(3, dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = target - cam_pos
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(world_up, forward)
    n = np.linalg.norm(right)
    if n < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= n
    up = np.cross(forward, right)
    up /= max(np.linalg.norm(up), 1e-9)
    return cam_pos, right, up, forward


def _prepare_orbit_cloud(
    points: np.ndarray,
    *,
    extent_m: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    xyz = points[:, :3].astype(np.float64)
    cen = np.median(xyz, axis=0)
    xyz = xyz - cen
    if extent_m is None:
        extent_m = float(np.percentile(np.linalg.norm(xyz, axis=1), 90))
        extent_m = max(extent_m, 2.5)
    return xyz, cen, extent_m


def _project_orbit(
    xyz: np.ndarray,
    *,
    yaw: float,
    extent_m: float,
    width: int,
    height: int,
    zenith_rad: float = _ISO_ZENITH_RAD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    cam_pos, right, up, forward = _look_at_downward_orbit(
        yaw, extent_m=extent_m, zenith_rad=zenith_rad
    )
    rel = xyz - cam_pos
    x_cam = rel @ right
    y_cam = rel @ up
    z_cam = np.maximum(rel @ forward, 1e-3)
    scale = (min(width, height) * 0.42) / extent_m
    px = width / 2.0 + x_cam * scale
    py = height / 2.0 - y_cam * scale
    return px, py, z_cam, scale


def _blit_point_overwrite(
    img: np.ndarray,
    px: float,
    py: float,
    kernel: np.ndarray,
    color: np.ndarray,
) -> None:
    h, w = img.shape[:2]
    kh, kw = kernel.shape
    ix = int(round(px)) - kw // 2
    iy = int(round(py)) - kh // 2
    x0, x1 = max(0, ix), min(w, ix + kw)
    y0, y1 = max(0, iy), min(h, iy + kh)
    kx0, ky0 = x0 - ix, y0 - iy
    kx1, ky1 = kx0 + (x1 - x0), ky0 + (y1 - y0)
    if x0 >= x1 or y0 >= y1:
        return
    k = kernel[ky0:ky1, kx0:kx1]
    m = k > 0.35
    for c in range(3):
        slab = img[y0:y1, x0:x1, c]
        slab[m] = color[c]
        img[y0:y1, x0:x1, c] = slab


def _blit_kernel(
    color_sum: np.ndarray,
    alpha: np.ndarray,
    px: float,
    py: float,
    kernel: np.ndarray,
    color: np.ndarray,
) -> None:
    """Accumulate premultiplied color; alpha excludes background."""
    h, w = color_sum.shape[:2]
    kh, kw = kernel.shape
    ix = int(round(px)) - kw // 2
    iy = int(round(py)) - kh // 2
    x0, x1 = max(0, ix), min(w, ix + kw)
    y0, y1 = max(0, iy), min(h, iy + kh)
    kx0, ky0 = x0 - ix, y0 - iy
    kx1, ky1 = kx0 + (x1 - x0), ky0 + (y1 - y0)
    if x0 >= x1 or y0 >= y1:
        return
    k = kernel[ky0:ky1, kx0:kx1]
    for c in range(3):
        color_sum[y0:y1, x0:x1, c] += k * color[c]
    alpha[y0:y1, x0:x1] += k


def _composite_accumulated(
    color_sum: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    h, w = color_sum.shape[:2]
    img = np.broadcast_to(_BG_RGB, (h, w, 3)).copy()
    m = alpha > _SPLAT_ALPHA_THRESHOLD
    if not np.any(m):
        return img
    a = np.clip(alpha[m], _SPLAT_ALPHA_THRESHOLD, 1.0)
    col = color_sum[m] / alpha[m, np.newaxis]
    wgt = a[:, np.newaxis] * col + (1.0 - a[:, np.newaxis]) * _BG_RGB
    img[m] = wgt
    return np.clip(img, 0.0, 1.0)


def _orbit_cloud_frame(
    xyz: np.ndarray,
    colors: np.ndarray,
    *,
    width: int,
    height: int,
    yaw: float,
    extent_m: float,
    style: str,
    zenith_rad: float = _ISO_ZENITH_RAD,
) -> np.ndarray:
    if len(xyz) == 0:
        return np.zeros((height, width, 3), dtype=np.float32)

    px, py, z_cam, _scale = _project_orbit(
        xyz, yaw=yaw, extent_m=extent_m, width=width, height=height, zenith_rad=zenith_rad
    )
    order = np.argsort(-z_cam)
    color_sum = np.zeros((height, width, 3), dtype=np.float32)
    alpha = np.zeros((height, width), dtype=np.float32)

    depth_span = float(z_cam.max() - z_cam.min() + 1e-6)
    if style == "height_points":
        img = np.broadcast_to(_BG_RGB, (height, width, 3)).copy()
        k = _kernel("disk", 1.65)
        for i in order:
            if px[i] < -4 or px[i] > width + 4 or py[i] < -4 or py[i] > height + 4:
                continue
            _blit_point_overwrite(img, px[i], py[i], k, colors[i])
        return np.clip(img, 0.0, 1.0)

    for i in order:
        if px[i] < -12 or px[i] > width + 12 or py[i] < -12 or py[i] > height + 12:
            continue
        depth_norm = (z_cam[i] - z_cam.min()) / depth_span
        sig = _SPLAT_SIGMA_MIN + _SPLAT_SIGMA_DEPTH * depth_norm
        k = _kernel("gauss", sig)
        _blit_kernel(color_sum, alpha, px[i], py[i], k, colors[i])
    return _composite_accumulated(color_sum, alpha)


def _load_showcase_crop(pick: ShowcasePick, crop_seed: int) -> np.ndarray | None:
    rng = np.random.default_rng(crop_seed)
    return extract_crop_at_global(
        global_x=pick.x,
        global_y=pick.y,
        in_channels=IN_CHANNELS_RGB,
        rng=rng,
    )


def render_cloud_orbit_frames(
    pick: ShowcasePick,
    *,
    n_frames: int = 120,
    rotations: float = 1.4,
    width: int = 960,
    height: int = 960,
    mode_blend: int = 10,
    mode_cycles: int = 2,
    crop_seed: int = 0,
    max_splat_points: int = 1400,
    max_point_points: int = 2200,
    on_frame: callable | None = None,
) -> list[Image.Image]:
    """Downward isometric orbit: alternate soft RGB splats and height-colored points."""
    rng = np.random.default_rng(crop_seed)
    crop = _load_showcase_crop(pick, crop_seed)
    if crop is None:
        blank = Image.new("RGB", (width, height), (15, 20, 25))
        return [blank] * n_frames

    orbit_cap = min(max_splat_points, max_point_points)
    if len(crop) > orbit_cap:
        idx = rng.choice(len(crop), size=orbit_cap, replace=False)
        crop = crop[idx]

    xyz, _cen, extent_m = _prepare_orbit_cloud(crop, extent_m=None)
    height_rgb = height_colors(xyz[:, 2])
    splat_rgb = np.clip(crop[:, 6:9], 0.0, 1.0).astype(np.float32)

    modes = (["rgb_splat", "height_points"] * mode_cycles)[: mode_cycles * 2]
    n_blocks = len(modes)
    blends = max(0, n_blocks - 1)
    total_blend = blends * mode_blend
    per_block = max(1, (n_frames - total_blend) // n_blocks)

    def _frame_schedule() -> list[tuple[str, str | None, float]]:
        schedule: list[tuple[str, str | None, float]] = []
        for bi, mode in enumerate(modes):
            schedule.extend((mode, None, 0.0) for _ in range(per_block))
            if bi < n_blocks - 1 and mode_blend > 0:
                nxt = modes[bi + 1]
                for j in range(mode_blend):
                    t = _ease_in_out((j + 1) / (mode_blend + 1))
                    schedule.append((mode, nxt, t))
        while len(schedule) < n_frames:
            schedule.append((modes[-1], None, 0.0))
        return schedule[:n_frames]

    schedule = _frame_schedule()

    def _yaw_at_index(frame_i: int) -> float:
        t = frame_i / max(n_frames - 1, 1)
        return t * rotations * 2.0 * math.pi

    def _render_buf(style: str, yaw: float) -> np.ndarray:
        colors = height_rgb if style == "height_points" else splat_rgb
        return _orbit_cloud_frame(
            xyz,
            colors,
            width=width,
            height=height,
            yaw=yaw,
            extent_m=extent_m,
            style=style,
        )

    out: list[Image.Image] = []
    for frame_i, (mode_a, mode_b, mix) in enumerate(schedule):
        yaw = _yaw_at_index(frame_i)
        if mode_b is None or mix <= 0.0:
            buf = _render_buf(mode_a, yaw)
            out.append(Image.fromarray((buf * 255).astype(np.uint8)))
            phase = mode_a
        else:
            a = _render_buf(mode_a, yaw)
            b = _render_buf(mode_b, yaw)
            buf = (1.0 - mix) * a + mix * b
            out.append(Image.fromarray((np.clip(buf, 0, 1) * 255).astype(np.uint8)))
            phase = f"{mode_a}>{mode_b}"
        if on_frame is not None:
            on_frame(frame_i + 1, n_frames, phase)
    return out


def render_splat_frames(
    pick: ShowcasePick,
    *,
    n_frames: int = 27,
    width: int = 960,
    height: int = 960,
    crop_seed: int = 0,
    on_frame: callable | None = None,
) -> list[Image.Image]:
    """Backward-compatible wrapper around the longer alternating cloud orbit."""
    return render_cloud_orbit_frames(
        pick,
        n_frames=n_frames,
        width=width,
        height=height,
        crop_seed=crop_seed,
        on_frame=on_frame,
    )


def crossfade_frames(
    a: list[Image.Image],
    b: list[Image.Image],
    n_blend: int,
) -> list[Image.Image]:
    if not a or not b or n_blend <= 0:
        return a + b
    out: list[Image.Image] = list(a)
    last = a[-1].convert("RGBA")
    first = b[0].convert("RGBA")
    w, h = last.size
    for i in range(1, n_blend + 1):
        t = i / (n_blend + 1)
        blend = Image.blend(last, first.resize((w, h)), t)
        out.append(blend.convert("RGB"))
    out.extend(b)
    return out


def dip_to_black(frames: list[Image.Image], n: int = 6) -> list[Image.Image]:
    if not frames or n <= 0:
        return frames
    w, h = frames[0].size
    black = Image.new("RGB", (w, h), (15, 20, 25))
    out = list(frames)
    for _ in range(n):
        out.append(black)
    return out


def load_register_lambert(parquet: Path, *, max_points: int = 6000) -> np.ndarray:
    import pandas as pd

    df = pd.read_parquet(parquet)
    if "eligible" in df.columns:
        df = df[df["eligible"]]
    if df.empty:
        return np.zeros((0, 2), dtype=np.float64)
    xy = df[["x", "y"]].to_numpy(dtype=np.float64)
    if len(xy) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xy), size=max_points, replace=False)
        xy = xy[idx]
    return xy


def city_bounds_from_register(
    reg: np.ndarray,
    pad_m: float = 500.0,
) -> tuple[float, float, float, float]:
    if len(reg) == 0:
        return (103000, 197000, 105000, 199000)
    return (
        float(reg[:, 0].min()) - pad_m,
        float(reg[:, 1].min()) - pad_m,
        float(reg[:, 0].max()) + pad_m,
        float(reg[:, 1].max()) + pad_m,
    )
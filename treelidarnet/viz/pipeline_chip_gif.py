"""Cinematic single-chip GIF: BEV → peaks → 0.85 gate → dual verifier → combo slider."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

from treelidarnet.data.chips import CELL_SIZE_M
from treelidarnet.data.crops import chip_id_for_point
from treelidarnet.data.crs import wgs84_to_lambert
from treelidarnet.datasets.chip_dataset import ChipDataset
from treelidarnet.explorer.combine import weighted_combo
from treelidarnet.infer import load_model
from treelidarnet.media.render_showcase import (
    ShowcasePick,
    crossfade_frames,
    render_cloud_orbit_frames,
)
from treelidarnet.media.showcase_select import chip_has_rgb
from treelidarnet.models.crop_verifier import load_crop_verifier
from treelidarnet.paths import chips_dir
from treelidarnet.postprocess import DEFAULT_PEAK_SCORE_THRESHOLD, Detection, extract_peaks_from_heatmap
from treelidarnet.verifier_batch import score_detections_dual_batched
def sweep_thresholds(low: float, high: float, step: float) -> list[float]:
    """Monotonic low → high (pipeline act 6; city combo GIF still bounces)."""
    return list(np.arange(low, high + step * 0.5, step))
from treelidarnet.viz.hsl import combo_colors, hsl_to_rgb

# Ghent city-center (Lambert 72); chip = floor(x/50)*50 via chip_id_for_point.
DEFAULT_CENTER_LON = 3.7219
DEFAULT_CENTER_LAT = 51.0543
DEFAULT_CENTER_X = 103869.12445620635
DEFAULT_CENTER_Y = 194243.87553793844

BG = "#0f1419"
FG = "#e7e9ea"
MUTED = "#8b949e"
ACCENT = "#fbbf24"
REJECT = "#6b7280"
FAIL = "#f87171"
PASS = "#34d399"
BAR_V2 = "#60a5fa"
BAR_V3 = "#34d399"


@dataclass
class ScoredPeak:
    x: float
    y: float
    score: float
    v2: float = 0.0
    v3: float = 0.0

    @property
    def combo(self) -> float:
        return weighted_combo(self.v2, self.v3)

    def passes_bev_pool(self, bev_min: float) -> bool:
        return self.score >= bev_min


@dataclass
class PipelineChipData:
    chip_id: str
    origin_x: float
    origin_y: float
    heat: np.ndarray
    peaks: list[ScoredPeak]
    bev_pool_min: float
    hero_index: int
    hard_neg: np.ndarray | None = None

    @property
    def grid_size(self) -> int:
        return int(self.heat.shape[0])

    def pool_peaks(self) -> list[ScoredPeak]:
        return [p for p in self.peaks if p.passes_bev_pool(self.bev_pool_min)]

    def hero(self) -> ScoredPeak:
        return self.peaks[self.hero_index]


def global_to_chip_px(
    x: float,
    y: float,
    *,
    origin_x: float,
    origin_y: float,
    cell_size: float = CELL_SIZE_M,
    grid_size: int,
) -> tuple[float, float]:
    """Grid indices matching extract_peaks_from_heatmap (row 0 = min y)."""
    del grid_size
    px = (x - origin_x) / cell_size - 0.5
    py = (y - origin_y) / cell_size - 0.5
    return px, py


def chip_px_to_axes_uv(px: float, py: float, grid_size: int) -> tuple[float, float]:
    """Matplotlib imshow(origin=upper): flip y."""
    return px / grid_size, 1.0 - py / grid_size


def _peak_uv(
    p: ScoredPeak, *, origin_x: float, origin_y: float, grid_size: int
) -> tuple[float, float]:
    px, py = global_to_chip_px(
        p.x, p.y, origin_x=origin_x, origin_y=origin_y, grid_size=grid_size
    )
    return chip_px_to_axes_uv(px, py, grid_size)


def _mark_rejected_peak(
    ax: plt.Axes,
    u: float,
    v: float,
    *,
    scale: float = 1.0,
    alpha: float = 0.85,
    zorder: int = 2,
) -> None:
    """Dim hollow ring + X — reads as culled / below threshold."""
    ax.scatter(
        [u],
        [v],
        s=90 * scale,
        facecolors="none",
        edgecolors=FAIL,
        linewidths=1.0,
        alpha=0.35 * alpha,
        zorder=zorder,
    )
    ax.scatter(
        [u],
        [v],
        marker="x",
        s=70 * scale,
        c=FAIL,
        linewidths=2.0,
        alpha=0.75 * alpha,
        zorder=zorder + 1,
    )


def _mark_shipped_peak(
    ax: plt.Axes,
    u: float,
    v: float,
    rgb: tuple[float, float, float] | np.ndarray,
    *,
    scale: float = 1.0,
    alpha: float = 1.0,
    zorder: int = 4,
    pulse: float = 0.0,
) -> None:
    col = np.asarray(rgb, dtype=np.float64).reshape(-1)[:3]
    pulse = float(np.clip(pulse, 0.0, 1.0))
    if pulse > 0.0:
        blob_scale = 0.75 + 0.55 * pulse
        ax.scatter(
            [u],
            [v],
            s=900 * scale * blob_scale,
            c=[col],
            alpha=(0.11 + 0.16 * pulse) * alpha,
            linewidths=0,
            zorder=zorder - 1,
        )
        ax.scatter(
            [u],
            [v],
            s=620 * scale * blob_scale,
            c="#ffffff",
            alpha=(0.06 + 0.10 * pulse) * alpha,
            linewidths=0,
            zorder=zorder,
        )
    ax.scatter(
        [u],
        [v],
        s=420 * scale * (1.0 + 0.08 * pulse),
        c="#ffffff",
        alpha=(0.14 + 0.08 * pulse) * alpha,
        linewidths=0,
        zorder=zorder + 1,
    )
    ax.scatter(
        [u],
        [v],
        s=200 * scale * (1.0 + 0.05 * pulse),
        c=[col],
        edgecolors="#ffffff",
        linewidths=2.0,
        alpha=min(1.0, alpha),
        zorder=zorder + 2,
    )


def detections_to_chip_px(
    peaks: list[ScoredPeak],
    *,
    origin_x: float,
    origin_y: float,
    grid_size: int,
) -> np.ndarray:
    out = np.empty((len(peaks), 2), dtype=np.float64)
    for i, p in enumerate(peaks):
        out[i] = global_to_chip_px(
            p.x, p.y, origin_x=origin_x, origin_y=origin_y, grid_size=grid_size
        )
    return out


def chip_id_at_lambert(x: float, y: float) -> str:
    return chip_id_for_point(float(x), float(y))


def chip_id_at_wgs84(lon: float, lat: float) -> str:
    x, y = wgs84_to_lambert(lon, lat)
    return chip_id_at_lambert(x, y)


def resolve_pipeline_chip_id(
    *,
    chip_id: str | None = None,
    lon: float | None = None,
    lat: float | None = None,
    x: float | None = None,
    y: float | None = None,
    require_rgb: bool = True,
    chip_root: Path | None = None,
) -> str:
    """Pick the 50 m chip containing a map point (not export-driven drama search)."""
    if chip_id is not None:
        cid = chip_id.strip()
    elif x is not None and y is not None:
        cid = chip_id_at_lambert(x, y)
    elif lon is not None and lat is not None:
        cid = chip_id_at_wgs84(lon, lat)
    else:
        cid = chip_id_at_lambert(DEFAULT_CENTER_X, DEFAULT_CENTER_Y)

    root = chip_root or chips_dir()
    npz = root / f"{cid}.npz"
    if not npz.is_file():
        raise FileNotFoundError(f"no chip npz for coordinate → {cid} ({npz})")
    if require_rgb and not chip_has_rgb(cid, chip_root=root):
        raise ValueError(f"chip {cid} at coordinate has no RGB — patch chips or pick another point")
    return cid


def prepare_chip_pipeline(
    *,
    chip_id: str,
    bev_checkpoint: Path,
    verifier_v2: Path,
    verifier_v3: Path,
    bev_pool_min: float = 0.85,
    peak_extract_min: float = DEFAULT_PEAK_SCORE_THRESHOLD,
    w_v2: float = 0.5,
    w_v3: float = 0.5,
    hero: str = "combo",
) -> PipelineChipData:
    npz = chips_dir() / f"{chip_id}.npz"
    if not npz.is_file():
        raise FileNotFoundError(npz)
    if not chip_has_rgb(chip_id):
        raise ValueError(f"chip {chip_id} missing RGB — pick another or patch chips")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(bev_checkpoint, device)
    ds = ChipDataset([npz], labels=False)
    batch = ds[0]
    bev = batch["bev"].unsqueeze(0).to(device)
    with torch.inference_mode():
        heat = model(bev).squeeze(0).squeeze(0).cpu().numpy()

    origin = batch["origin"]
    x0, y0 = float(origin[0]), float(origin[1])

    dets = extract_peaks_from_heatmap(
        heat,
        origin_x=x0,
        origin_y=y0,
        cell_size=CELL_SIZE_M,
        score_threshold=peak_extract_min,
    )
    all_peaks = [ScoredPeak(x=d.x, y=d.y, score=d.score) for d in dets]
    pool = [Detection(x=p.x, y=p.y, score=p.score) for p in all_peaks if p.score >= bev_pool_min]

    v2_map: dict[tuple[float, float], float] = {}
    v3_map: dict[tuple[float, float], float] = {}
    if pool:
        mv2 = load_crop_verifier(verifier_v2, device)
        mv3 = load_crop_verifier(verifier_v3, device)
        _, dual = score_detections_dual_batched(
            pool,
            mv2,
            mv3,
            device=device,
        )
        for det, sc in zip(pool, dual, strict=True):
            key = (det.x, det.y)
            v2_map[key] = float(sc["v2"])
            v3_map[key] = float(sc["v3"])

    for p in all_peaks:
        key = (p.x, p.y)
        if key in v2_map:
            p.v2 = v2_map[key]
            p.v3 = v3_map[key]

    hard_neg = None
    with np.load(npz) as data:
        if "hard_negative" in data.files:
            hard_neg = data["hard_negative"].astype(np.float32)

    pool_scored = [p for p in all_peaks if p.passes_bev_pool(bev_pool_min) and p.v2 > 0]
    if pool_scored:
        if hero == "bev":
            hero_i = max(range(len(all_peaks)), key=lambda i: all_peaks[i].score)
        else:
            hero_i = max(
                range(len(all_peaks)),
                key=lambda i: all_peaks[i].combo if all_peaks[i].v2 > 0 else -1.0,
            )
    else:
        hero_i = 0 if all_peaks else 0

    return PipelineChipData(
        chip_id=chip_id,
        origin_x=x0,
        origin_y=y0,
        heat=heat,
        peaks=all_peaks,
        bev_pool_min=bev_pool_min,
        hero_index=hero_i,
        hard_neg=hard_neg,
    )


def _fig_to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buf[:, :, :3])


def _new_ax(figsize: tuple[float, float], dpi: int) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG, dpi=dpi)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _figure_with_hud(
    figsize: tuple[float, float], dpi: int
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Top HUD for captions; chip map uses the lower band only (peaks stay below text)."""
    fig = plt.figure(figsize=figsize, facecolor=BG, dpi=dpi)
    hud_h = 0.17
    chip_h = 1.0 - hud_h - 0.03
    left, w = 0.04, 0.92
    chip_ax = fig.add_axes([left, 0.03, w, chip_h])
    hud_ax = fig.add_axes([left, 0.03 + chip_h + 0.01, w, hud_h - 0.01])
    for ax in (chip_ax, hud_ax):
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    chip_ax.set_aspect("equal", adjustable="box")
    return fig, chip_ax, hud_ax


def _chip_rgba_from_npz(chip_id: str, grid_size: int) -> np.ndarray | None:
    npz = chips_dir() / f"{chip_id}.npz"
    if not npz.is_file():
        return None
    with np.load(npz) as data:
        xyz = data["xyz"]
        if len(xyz) == 0:
            return None
        z = xyz[:, 2]
        ix = np.clip((xyz[:, 0] / CELL_SIZE_M).astype(np.int32), 0, grid_size - 1)
        iy = np.clip((xyz[:, 1] / CELL_SIZE_M).astype(np.int32), 0, grid_size - 1)
        grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
        for i in range(len(z)):
            gy, gx = iy[i], ix[i]
            prev = grid[gy, gx]
            if math.isnan(prev) or z[i] > prev:
                grid[gy, gx] = z[i]
        dens = np.isfinite(grid).astype(np.float32)
        z_norm = grid.copy()
        valid = np.isfinite(z_norm)
        if valid.any():
            z_min, z_max = float(z_norm[valid].min()), float(z_norm[valid].max())
            z_norm[~valid] = z_min
            if z_max > z_min:
                z_norm = (z_norm - z_min) / (z_max - z_min)
            else:
                z_norm = np.zeros_like(z_norm)
        else:
            z_norm = np.zeros_like(grid)
        cmap = plt.get_cmap("viridis")
        rgba = cmap(z_norm)
        rgba[..., 3] = np.clip(dens * 0.85 + 0.08, 0, 1)
        return rgba


def _draw_chip_panel(
    ax: plt.Axes,
    data: PipelineChipData,
    *,
    alpha_heat: float = 0.0,
    peaks: list[ScoredPeak] | None = None,
    peak_alpha: float = 1.0,
    gate_thr: float | None = None,
    color_by_combo: bool = False,
    combo_thr: float | None = None,
    w_v2: float = 0.5,
    w_v3: float = 0.5,
    show_labels: bool = False,
    max_labels: int = 6,
) -> None:
    gs = data.grid_size
    rgba = _chip_rgba_from_npz(data.chip_id, gs)
    if rgba is not None and alpha_heat <= 0:
        ax.imshow(rgba, extent=[0, 1, 0, 1], origin="upper", aspect="auto")

    if alpha_heat > 0:
        norm = Normalize(vmin=0, vmax=1)
        hm = ax.imshow(
            data.heat,
            cmap="inferno",
            norm=norm,
            extent=[0, 1, 0, 1],
            origin="upper",
            aspect="auto",
            alpha=alpha_heat,
        )
        if alpha_heat >= 0.65 and data.hard_neg is not None and data.hard_neg.size == data.heat.size:
            mask = data.hard_neg > 0.5
            outline = np.ma.masked_where(~mask, mask.astype(float))
            ax.imshow(
                outline,
                cmap="Greys",
                alpha=0.22,
                extent=[0, 1, 0, 1],
                origin="upper",
                aspect="auto",
                vmin=0,
                vmax=1,
            )

    use = peaks if peaks is not None else data.peaks
    if not use:
        return

    if color_by_combo and combo_thr is not None:
        pool = [p for p in use if p.passes_bev_pool(data.bev_pool_min)]
        if pool:
            v2 = np.array([p.v2 for p in pool])
            v3 = np.array([p.v3 for p in pool])
            rgb = combo_colors(v2, v3, w_v2=w_v2, w_v3=w_v3)
            for i, p in enumerate(pool):
                u, v = _peak_uv(p, origin_x=data.origin_x, origin_y=data.origin_y, grid_size=gs)
                if p.combo < combo_thr:
                    _mark_rejected_peak(ax, u, v, scale=0.55, alpha=peak_alpha)
                else:
                    _mark_shipped_peak(ax, u, v, rgb[i] / 255.0, scale=0.65, alpha=peak_alpha)
        return

    labeled = 0
    for p in use:
        px, py = global_to_chip_px(
            p.x, p.y, origin_x=data.origin_x, origin_y=data.origin_y, grid_size=gs
        )
        u, v = chip_px_to_axes_uv(px, py, gs)
        passes = gate_thr is None or p.score >= gate_thr
        pool = p.passes_bev_pool(data.bev_pool_min)
        if gate_thr is not None and not passes:
            _mark_rejected_peak(ax, u, v, scale=0.5, alpha=peak_alpha * 0.9, zorder=3)
            continue
        elif gate_thr is not None and passes and pool:
            ax.scatter(
                [u],
                [v],
                s=100,
                c=PASS,
                edgecolors="#ffffff",
                linewidths=1.8,
                alpha=peak_alpha,
                zorder=4,
            )
        elif gate_thr is not None and passes:
            ax.scatter(
                [u],
                [v],
                s=55,
                c=ACCENT,
                edgecolors="#ffffff",
                linewidths=0.8,
                alpha=peak_alpha * 0.6,
                zorder=3,
            )
        else:
            ax.scatter(
                [u],
                [v],
                s=50,
                c=ACCENT,
                edgecolors="#fff",
                linewidths=0.5,
                alpha=peak_alpha,
                zorder=3,
            )
        if show_labels and labeled < max_labels:
            ax.text(
                u + 0.02,
                v,
                f"{p.score:.2f}",
                color=FG,
                fontsize=9,
                va="center",
                alpha=al,
            )
            labeled += 1


def _draw_export_finale(
    ax: plt.Axes,
    data: PipelineChipData,
    *,
    export_thr: float,
    w_v2: float,
    w_v3: float,
    reveal: float = 1.0,
    pulse: float = 0.0,
) -> int:
    """Crisp end card: terrain + large combo-colored crowns for shipped detections."""
    gs = data.grid_size
    rgba = _chip_rgba_from_npz(data.chip_id, gs)
    if rgba is not None:
        bg = rgba.copy()
        bg[..., 3] = 1.0
        ax.imshow(bg, extent=[0, 1, 0, 1], origin="upper", aspect="auto", zorder=0)
    ax.imshow(
        data.heat,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=[0, 1, 0, 1],
        origin="upper",
        aspect="auto",
        alpha=0.12 * reveal,
        zorder=1,
    )

    pool = data.pool_peaks()
    kept = [p for p in pool if p.combo >= export_thr]
    dropped = [p for p in pool if p.combo < export_thr]

    for p in dropped:
        u, v = _peak_uv(p, origin_x=data.origin_x, origin_y=data.origin_y, grid_size=gs)
        _mark_rejected_peak(ax, u, v, scale=0.85, alpha=reveal, zorder=2)

    if not kept:
        return 0

    v2 = np.array([p.v2 for p in kept], dtype=np.float64)
    v3 = np.array([p.v3 for p in kept], dtype=np.float64)
    rgb = combo_colors(v2, v3, w_v2=w_v2, w_v3=w_v3)
    for i, p in enumerate(kept):
        u, v = _peak_uv(p, origin_x=data.origin_x, origin_y=data.origin_y, grid_size=gs)
        _mark_shipped_peak(
            ax,
            u,
            v,
            rgb[i] / 255.0,
            scale=reveal,
            alpha=min(1.0, 0.4 + 0.6 * reveal),
            pulse=pulse * reveal,
        )
        label_a = min(1.0, 0.35 + 0.65 * reveal)
        ax.text(
            u,
            v - 0.032,
            f"{p.combo:.2f}",
            ha="center",
            va="top",
            fontsize=7,
            color=MUTED,
            alpha=label_a,
            zorder=8,
        )
    return len(kept)


def _title_card(
    data: PipelineChipData,
    *,
    figsize: tuple[float, float],
    dpi: int,
    subtitle: str,
) -> Image.Image:
    fig, ax = _new_ax(figsize, dpi)
    ax.text(0.5, 0.58, "TreeLidarNet", ha="center", color=FG, fontsize=28, fontweight="bold")
    ax.text(0.5, 0.44, "One 50 m chip · full pipeline", ha="center", color=MUTED, fontsize=15)
    ax.text(0.5, 0.32, data.chip_id.replace("_", " "), ha="center", color=ACCENT, fontsize=13)
    ax.text(0.5, 0.22, subtitle, ha="center", color=MUTED, fontsize=11)
    ax.text(0.5, 0.1, "Flanders DHMV II LiDAR · Ghent", ha="center", color=MUTED, fontsize=10)
    return _fig_to_image(fig)


def _info_strip(hud_ax: plt.Axes, line1: str, line2: str = "", *, extra: str = "") -> None:
    hud_ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.012",
            facecolor="#161b22",
            edgecolor="#30363d",
            linewidth=0.8,
            transform=hud_ax.transAxes,
            zorder=5,
        )
    )
    hud_ax.text(
        0.03,
        0.68,
        line1,
        transform=hud_ax.transAxes,
        color=FG,
        fontsize=12,
        fontweight="bold",
        va="center",
        zorder=10,
    )
    if line2:
        hud_ax.text(
            0.03,
            0.34,
            line2,
            transform=hud_ax.transAxes,
            color=MUTED,
            fontsize=10,
            va="center",
            zorder=10,
        )
    if extra:
        hud_ax.text(
            0.97,
            0.5,
            extra,
            transform=hud_ax.transAxes,
            color=FG,
            fontsize=11,
            ha="right",
            va="center",
            zorder=10,
        )


def _hud_score_bars(hud_ax: plt.Axes, hero: ScoredPeak) -> None:
    x0 = 0.52
    y = 0.72
    for label, val, col in (
        ("v2", hero.v2, BAR_V2),
        ("v3", hero.v3, BAR_V3),
        ("combo", hero.combo, ACCENT),
    ):
        hud_ax.add_patch(
            Rectangle((x0, y - 0.22), 0.32, 0.08, transform=hud_ax.transAxes, facecolor="#161b22", zorder=6)
        )
        hud_ax.add_patch(
            Rectangle(
                (x0, y - 0.22),
                0.32 * min(1.0, val),
                0.08,
                transform=hud_ax.transAxes,
                facecolor=col,
                zorder=7,
            )
        )
        hud_ax.text(
            x0 + 0.34,
            y - 0.18,
            f"{label} {val:.2f}",
            transform=hud_ax.transAxes,
            color=FG,
            fontsize=9,
            va="center",
            zorder=10,
        )
        y -= 0.28


def _hud_bev_slider(hud_ax: plt.Axes, thr: float) -> None:
    hud_ax.add_patch(
        Rectangle((0.52, 0.12), 0.44, 0.12, transform=hud_ax.transAxes, facecolor="#161b22", zorder=6)
    )
    hud_ax.add_patch(
        Rectangle((0.52, 0.12), 0.44 * thr, 0.12, transform=hud_ax.transAxes, facecolor=ACCENT, zorder=7)
    )
    hud_ax.text(
        0.74,
        0.18,
        f"BEV ≥ {thr:.2f}",
        ha="center",
        color=FG,
        fontsize=10,
        transform=hud_ax.transAxes,
        zorder=10,
    )


def _render_act_frames(
    data: PipelineChipData,
    *,
    width: int,
    height: int,
    w_v2: float,
    w_v3: float,
    combo_low: float = 0.45,
    combo_high: float = 0.7,
    combo_step: float = 0.02,
) -> tuple[list[Image.Image], list[int]]:
    dpi = 100
    figsize = (width / dpi, height / dpi)
    frames: list[Image.Image] = []

    for _ in range(10):
        frames.append(_title_card(data, figsize=figsize, dpi=dpi, subtitle="LiDAR → BEV → verifier → export"))

    for i in range(12):
        t = i / 11
        fig, chip_ax, hud_ax = _figure_with_hud(figsize, dpi)
        gs = data.grid_size
        rgba = _chip_rgba_from_npz(data.chip_id, gs)
        if rgba is not None:
            rgba2 = rgba.copy()
            rgba2[..., 3] *= t
            chip_ax.imshow(rgba2, extent=[0, 1, 0, 1], origin="upper", aspect="auto")
        _info_strip(hud_ax, "Act 1 · Input", "Max height + density BEV from point cloud")
        frames.append(_fig_to_image(fig))

    for i in range(18):
        t = i / 17
        fig, chip_ax, hud_ax = _figure_with_hud(figsize, dpi)
        _draw_chip_panel(chip_ax, data, alpha_heat=t)
        _info_strip(hud_ax, "Act 2 · BEV U-Net", "Canopy heatmap (frozen gent_canopy_v2)")
        frames.append(_fig_to_image(fig))

    n_peaks = len(data.peaks)
    for i in range(16):
        shown = max(0, int(round((i + 1) / 16 * n_peaks))) if n_peaks else 0
        fig, chip_ax, hud_ax = _figure_with_hud(figsize, dpi)
        _draw_chip_panel(chip_ax, data, alpha_heat=0.92, peaks=data.peaks[:shown], peak_alpha=1.0)
        _info_strip(
            hud_ax,
            "Act 3 · Peak extract",
            f"Local maxima @ confidence ≥ {DEFAULT_PEAK_SCORE_THRESHOLD:.2f} · {n_peaks} peaks",
        )
        frames.append(_fig_to_image(fig))

    for i in range(22):
        thr = 0.20 + (0.85 - 0.20) * (i / 21)
        pool_n = sum(1 for p in data.peaks if p.score >= thr)
        scored_n = sum(1 for p in data.peaks if p.score >= data.bev_pool_min)
        fig, chip_ax, hud_ax = _figure_with_hud(figsize, dpi)
        _draw_chip_panel(chip_ax, data, alpha_heat=0.75, gate_thr=thr, peak_alpha=1.0)
        _hud_bev_slider(hud_ax, thr)
        _info_strip(
            hud_ax,
            "Act 4 · Verifier pool gate",
            f"{scored_n} peaks ≥ {data.bev_pool_min:.2f} → dual crop score  ·  {pool_n} visible @ {thr:.2f}",
        )
        frames.append(_fig_to_image(fig))

    hero = data.hero()
    pick = ShowcasePick(
        lon=0.0,
        lat=0.0,
        x=hero.x,
        y=hero.y,
        bev_score=hero.score,
        verifier_score=hero.combo,
        chip_id=data.chip_id,
    )
    orbit = render_cloud_orbit_frames(
        pick,
        n_frames=20,
        rotations=0.35,
        width=320,
        height=320,
        mode_blend=0,
        mode_cycles=1,
    )

    pool = data.pool_peaks()
    for i, orb in enumerate(orbit):
        fig, chip_ax, hud_ax = _figure_with_hud(figsize, dpi)
        _draw_chip_panel(chip_ax, data, alpha_heat=0.55, gate_thr=data.bev_pool_min, peak_alpha=1.0)
        chip_ax.imshow(
            np.array(orb),
            extent=[0.62, 0.98, 0.52, 0.98],
            origin="upper",
            aspect="auto",
            zorder=4,
        )
        chip_ax.add_patch(
            Rectangle(
                (0.61, 0.51),
                0.38,
                0.48,
                transform=chip_ax.transAxes,
                fill=False,
                edgecolor=MUTED,
                linewidth=0.8,
                zorder=5,
            )
        )
        _hud_score_bars(hud_ax, hero)
        _info_strip(
            hud_ax,
            "Act 5 · Dual PointNet++ verifier",
            f"{w_v2:.1f}·v2 + {w_v3:.1f}·v3 on 512-pt crop · {len(pool)} scored peaks",
        )
        frames.append(_fig_to_image(fig))

    thresholds = sweep_thresholds(combo_low, combo_high, combo_step)
    for thr in thresholds:
        kept = sum(1 for p in pool if p.combo >= thr)
        fig, chip_ax, hud_ax = _figure_with_hud(figsize, dpi)
        _draw_chip_panel(
            chip_ax,
            data,
            alpha_heat=0.45,
            color_by_combo=True,
            combo_thr=thr,
            w_v2=w_v2,
            w_v3=w_v3,
        )
        t = (thr - combo_low) / max(combo_high - combo_low, 1e-6)
        hue = hsl_to_rgb(142 * t, 0.38, 0.52)
        hue_css = (hue[0] / 255, hue[1] / 255, hue[2] / 255)
        _info_strip(
            hud_ax,
            "Act 6 · Weighted export",
            "Green crowns = kept · red ✕ = below combo-t",
        )
        hud_ax.text(
            0.97,
            0.5,
            f"min combo {thr:.2f}  ·  {kept}/{len(pool)} kept",
            transform=hud_ax.transAxes,
            color=hue_css,
            fontsize=11,
            ha="right",
            va="center",
            zorder=11,
        )
        frames.append(_fig_to_image(fig))

    act7_start = len(frames)
    finale_thr = combo_high

    finale_frames = 30
    for i in range(finale_frames):
        reveal = min(1.0, (i + 1) / 10.0)
        phase = (i / max(finale_frames - 1, 1)) * math.pi * 4
        pulse = 0.5 + 0.5 * math.sin(phase)
        fig, chip_ax, hud_ax = _figure_with_hud(figsize, dpi)
        n_shown = _draw_export_finale(
            chip_ax,
            data,
            export_thr=finale_thr,
            w_v2=w_v2,
            w_v3=w_v3,
            reveal=reveal,
            pulse=pulse,
        )
        _info_strip(
            hud_ax,
            "Finale · tree product",
            f"{n_shown} at combo ≥ {finale_thr:.2f} (pulsing) · {len(pool) - n_shown} culled (✕)"
            f"  · export floor {combo_low:.2f}",
        )
        hud_ax.text(
            0.97,
            0.5,
            f"weighted 0.5·v2 + 0.5·v3",
            transform=hud_ax.transAxes,
            color=MUTED,
            fontsize=10,
            ha="right",
            va="center",
            zorder=11,
        )
        frames.append(_fig_to_image(fig))

    bounds = [0, 10, 22, 40, 56, 78, 98, act7_start, len(frames)]
    return frames, bounds


def render_pipeline_chip_gif(
    *,
    chip_id: str | None,
    out_path: Path,
    bev_checkpoint: Path,
    verifier_v2: Path,
    verifier_v3: Path,
    lon: float | None = None,
    lat: float | None = None,
    x: float | None = None,
    y: float | None = None,
    bev_pool_min: float = 0.85,
    width: int = 1280,
    height: int = 720,
    duration_ms: int = 140,
    crossfade: int = 4,
    w_v2: float = 0.5,
    w_v3: float = 0.5,
    hero: str = "combo",
    combo_low: float = 0.45,
    combo_high: float = 0.7,
    combo_step: float = 0.02,
) -> int:
    cid = resolve_pipeline_chip_id(
        chip_id=chip_id,
        lon=lon,
        lat=lat,
        x=x,
        y=y,
    )
    if chip_id is None:
        where = (
            f"lon/lat {lon},{lat}"
            if lon is not None and lat is not None
            else f"xy {x},{y}" if x is not None and y is not None
            else "Ghent center"
        )
        print(f"Chip at {where}: {cid}", flush=True)

    data = prepare_chip_pipeline(
        chip_id=cid,
        bev_checkpoint=bev_checkpoint,
        verifier_v2=verifier_v2,
        verifier_v3=verifier_v3,
        bev_pool_min=bev_pool_min,
        w_v2=w_v2,
        w_v3=w_v3,
        hero=hero,
    )

    raw, bounds = _render_act_frames(
        data,
        width=width,
        height=height,
        w_v2=w_v2,
        w_v3=w_v3,
        combo_low=combo_low,
        combo_high=combo_high,
        combo_step=combo_step,
    )

    segments: list[list[Image.Image]] = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        segments.append([f.resize((width, height)) for f in raw[a:b]])

    stitched: list[Image.Image] = segments[0]
    for seg in segments[1:]:
        stitched = crossfade_frames(stitched, seg, crossfade)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stitched[0].save(
        out_path,
        save_all=True,
        append_images=stitched[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(
        f"Wrote {out_path} ({len(stitched)} frames, chip={cid}, "
        f"{len(data.peaks)} peaks, {len(data.pool_peaks())} verifier pool)",
        flush=True,
    )
    return len(stitched)
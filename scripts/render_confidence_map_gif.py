#!/usr/bin/env python3
"""Animated Ghent pred map: min confidence sweeps 20%–40% and back (from city export JSON)."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from treelidarnet.data.crs import wgs84_to_lambert
from treelidarnet.paths import exports_dir, repo_root


def load_export(export_gz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with gzip.open(export_gz, "rt", encoding="utf-8") as f:
        data = json.load(f)
    preds = data["predictions"]
    xs, ys, scores = [], [], []
    for p in preds:
        lon, lat = p["position"]
        x, y = wgs84_to_lambert(lon, lat)
        xs.append(x)
        ys.append(y)
        scores.append(float(p["score"]))
    return np.asarray(xs), np.asarray(ys), np.asarray(scores), data


def bounce_thresholds(low: float, high: float, step: float) -> list[float]:
    up = list(np.arange(low, high + step * 0.5, step))
    down = list(np.arange(high - step, low - step * 0.5, -step))
    return up + down


def render_gif(
    *,
    export_gz: Path,
    out_path: Path,
    low: float = 0.20,
    high: float = 0.40,
    step: float = 0.02,
    dpi: int = 120,
    figsize: tuple[float, float] = (10, 10),
    duration_ms: int = 180,
    point_size: float = 0.35,
    max_points: int | None = None,
) -> None:
    xs, ys, scores, meta = load_export(export_gz)
    if max_points and len(xs) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xs), size=max_points, replace=False)
        xs, ys, scores = xs[idx], ys[idx], scores[idx]

    thresholds = bounce_thresholds(low, high, step)
    pad = 400.0
    xmin, xmax = xs.min() - pad, xs.max() + pad
    ymin, ymax = ys.min() - pad, ys.max() + pad

    frames: list[Image.Image] = []
    model_id = meta.get("model_id", "?")

    for thr in thresholds:
        mask = scores >= thr
        n = int(mask.sum())
        fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1419")
        ax.set_facecolor("#0f1419")
        ax.scatter(
            xs[mask],
            ys[mask],
            s=point_size,
            c="#f59e0b",
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        pct = int(round(thr * 100))
        ax.set_title(
            f"Ghent tree detections — min confidence {pct}%\n"
            f"{model_id} · {n:,} shown · export floor ≥{int(low*100)}%",
            color="#e7e9ea",
            fontsize=14,
            pad=12,
        )
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf))
        plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Wrote {out_path} ({len(frames)} frames, {duration_ms} ms/frame)")


def main() -> None:
    default_in = exports_dir() / "inference" / "gent_city_gent_canopy_v2.json.gz"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=default_in)
    p.add_argument(
        "--output",
        type=Path,
        default=exports_dir() / "inference" / "gent_confidence_20_40_loop.gif",
    )
    p.add_argument("--low", type=float, default=0.20)
    p.add_argument("--high", type=float, default=0.40)
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--duration-ms", type=int, default=180)
    args = p.parse_args()
    inp = args.input if args.input.is_absolute() else repo_root() / args.input
    out = args.output if args.output.is_absolute() else repo_root() / args.output
    render_gif(
        export_gz=inp,
        out_path=out,
        low=args.low,
        high=args.high,
        step=args.step,
        duration_ms=args.duration_ms,
    )


if __name__ == "__main__":
    main()

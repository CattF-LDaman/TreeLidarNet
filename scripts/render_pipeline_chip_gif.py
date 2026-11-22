#!/usr/bin/env python3
"""Cinematic GIF: one chip — BEV → 0.85 gate → dual verifier → combo slider."""

from __future__ import annotations

import argparse
from pathlib import Path

from treelidarnet.paths import exports_dir, repo_root
from treelidarnet.viz.pipeline_chip_gif import (
    DEFAULT_CENTER_LAT,
    DEFAULT_CENTER_LON,
    render_pipeline_chip_gif,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chip-id", default=None, help="explicit gent_50m_X_Y")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--lon", type=float, default=None, help="WGS84 (use with --lat)")
    g.add_argument("--x", type=float, default=None, help="Lambert72 x (use with --y)")
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--y", type=float, default=None)
    p.add_argument(
        "--use-ghent-center",
        action="store_true",
        help=f"Default when no chip/coords: ({DEFAULT_CENTER_LON:.4f}, {DEFAULT_CENTER_LAT:.4f})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default exports/inference/pipeline_chip_<id>_loop.gif",
    )
    p.add_argument("--checkpoint", type=Path, default=Path("runs/gent_canopy_v2/best.pt"))
    p.add_argument("--verifier-v2", type=Path, default=Path("runs/gent_crop_verifier_v2_fp2/best.pt"))
    p.add_argument("--verifier-v3", type=Path, default=Path("runs/gent_crop_verifier_v3_rgb_fp2/best.pt"))
    p.add_argument("--bev-pool", type=float, default=0.85)
    p.add_argument("--combo-low", type=float, default=0.45)
    p.add_argument("--combo-high", type=float, default=0.7)
    p.add_argument("--combo-step", type=float, default=0.02)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--duration-ms", type=int, default=140)
    p.add_argument("--hero", choices=("combo", "bev"), default="combo")
    p.add_argument("--w-v2", type=float, default=0.5)
    p.add_argument("--w-v3", type=float, default=0.5)
    args = p.parse_args()

    lon, lat, x, y = args.lon, args.lat, args.x, args.y
    if args.chip_id is None and lon is None and x is None:
        lon, lat = DEFAULT_CENTER_LON, DEFAULT_CENTER_LAT

    if lon is not None and lat is None:
        p.error("--lat required with --lon")
    if lat is not None and lon is None:
        p.error("--lon required with --lat")
    if x is not None and y is None:
        p.error("--y required with --x")
    if y is not None and x is None:
        p.error("--x required with --y")

    def _resolve(pth: Path) -> Path:
        return pth if pth.is_absolute() else repo_root() / pth

    out = args.output
    if out is None:
        cid = args.chip_id or "coord"
        out = exports_dir() / "inference" / f"pipeline_chip_{cid}_loop.gif"
    else:
        out = _resolve(out)

    render_pipeline_chip_gif(
        chip_id=args.chip_id,
        out_path=out,
        bev_checkpoint=_resolve(args.checkpoint),
        verifier_v2=_resolve(args.verifier_v2),
        verifier_v3=_resolve(args.verifier_v3),
        lon=lon,
        lat=lat,
        x=x,
        y=y,
        bev_pool_min=args.bev_pool,
        width=args.width,
        height=args.height,
        duration_ms=args.duration_ms,
        w_v2=args.w_v2,
        w_v3=args.w_v3,
        hero=args.hero,
        combo_low=args.combo_low,
        combo_high=args.combo_high,
        combo_step=args.combo_step,
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Static map: weighted combo color at a single combo threshold (default export floor 0.45)."""

from __future__ import annotations

import argparse
from pathlib import Path

from treelidarnet.paths import exports_dir, repo_root
from treelidarnet.viz.combo_export_gif import render_combo_export_static


def main() -> None:
    default_in = exports_dir() / "inference" / "gent_city_gent_canopy_v2_verified_v2_v3_full.json.gz"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=default_in)
    p.add_argument(
        "--output",
        type=Path,
        default=exports_dir() / "inference" / "gent_combo_export_floor_0.45_wallpaper.png",
    )
    p.add_argument("--threshold", type=float, default=0.45, help="min combo (export floor)")
    p.add_argument("--width", type=int, default=3840, help="wallpaper width (16:9 with --height 2160)")
    p.add_argument("--height", type=int, default=2160)
    p.add_argument(
        "--zoom",
        type=float,
        default=3.0,
        help=">1 crops to median-centered core (narrower map = sharper trees)",
    )
    p.add_argument(
        "--point-radius",
        type=int,
        default=1,
        help="pixel splat radius (0=1px); 1 fills gaps when zoomed",
    )
    p.add_argument("--hue-low", type=float, default=0.0)
    p.add_argument("--hue-high", type=float, default=142.0)
    p.add_argument("--sat", type=float, default=0.38)
    p.add_argument("--light", type=float, default=0.52)
    p.add_argument("--w-v2", type=float, default=0.5)
    p.add_argument("--w-v3", type=float, default=0.5)
    args = p.parse_args()
    inp = args.input if args.input.is_absolute() else repo_root() / args.input
    out = args.output if args.output.is_absolute() else repo_root() / args.output
    render_combo_export_static(
        export_gz=inp,
        out_path=out,
        threshold=args.threshold,
        width=args.width,
        height=args.height,
        zoom=args.zoom,
        point_radius=args.point_radius,
        hue_low=args.hue_low,
        hue_high=args.hue_high,
        sat=args.sat,
        light=args.light,
        w_v2=args.w_v2,
        w_v3=args.w_v3,
    )


if __name__ == "__main__":
    main()
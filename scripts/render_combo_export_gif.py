#!/usr/bin/env python3
"""Animated map: weighted combo color + bouncing combo threshold on verified export."""

from __future__ import annotations

import argparse
from pathlib import Path

from treelidarnet.paths import exports_dir, repo_root
from treelidarnet.viz.combo_export_gif import render_combo_export_gif


def main() -> None:
    default_in = exports_dir() / "inference" / "gent_city_gent_canopy_v2_verified_v2_v3_full.json.gz"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=default_in)
    p.add_argument(
        "--output",
        type=Path,
        default=exports_dir() / "inference" / "gent_combo_045_100_loop.gif",
    )
    p.add_argument("--low", type=float, default=0.45, help="combo threshold low (export floor)")
    p.add_argument("--high", type=float, default=1.0)
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--duration-ms", type=int, default=160)
    p.add_argument("--width", type=int, default=2400, help="raster canvas pixels (square)")
    p.add_argument("--hue-low", type=float, default=0.0)
    p.add_argument("--hue-high", type=float, default=142.0)
    p.add_argument("--sat", type=float, default=0.38)
    p.add_argument("--light", type=float, default=0.52)
    p.add_argument("--w-v2", type=float, default=0.5)
    p.add_argument("--w-v3", type=float, default=0.5)
    args = p.parse_args()
    inp = args.input if args.input.is_absolute() else repo_root() / args.input
    out = args.output if args.output.is_absolute() else repo_root() / args.output
    render_combo_export_gif(
        export_gz=inp,
        out_path=out,
        low=args.low,
        high=args.high,
        step=args.step,
        duration_ms=args.duration_ms,
        canvas_px=args.width,
        hue_low=args.hue_low,
        hue_high=args.hue_high,
        sat=args.sat,
        light=args.light,
        w_v2=args.w_v2,
        w_v3=args.w_v3,
    )


if __name__ == "__main__":
    main()
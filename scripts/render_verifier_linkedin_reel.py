#!/usr/bin/env python3
"""LinkedIn MP4: map zoom + RGB splat per high-confidence verifier detection."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from treelidarnet.media.render_showcase import (
    city_bounds_from_register,
    crossfade_frames,
    dip_to_black,
    load_register_lambert,
    render_map_zoom_frames,
    render_cloud_orbit_frames,
)
from treelidarnet.media.showcase_select import select_showcase_picks
from treelidarnet.paths import exports_dir, manifests_dir, repo_root


def encode_mp4(frame_dir: Path, n_frames: int, out_mp4: Path, *, fps: int = 30) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH — install ffmpeg to write MP4")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "frame_%06d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-frames:v",
        str(n_frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def build_reel(
    *,
    verified_path: Path,
    output_mp4: Path,
    manifest_path: Path,
    trees_parquet: Path,
    n: int,
    bev_min: float,
    verifier_min: float,
    min_spacing_m: float,
    map_frames: int,
    cloud_frames: int,
    cloud_rotations: float,
    cloud_mode_blend: int,
    cloud_mode_cycles: int,
    blend_frames: int,
    black_frames: int,
    fps: int,
) -> None:
    picks = select_showcase_picks(
        verified_path,
        n=n,
        bev_min=bev_min,
        verifier_min=verifier_min,
        min_spacing_m=min_spacing_m,
    )
    if not picks:
        raise RuntimeError("No showcase picks — lower thresholds or finish RGB patch")

    reg = load_register_lambert(trees_parquet)
    bounds = city_bounds_from_register(reg)

    manifest = {"picks": [], "fps": fps, "verified": str(verified_path)}

    with tempfile.TemporaryDirectory(prefix="verifier_reel_") as tmp:
        tmp_path = Path(tmp)
        frame_idx = 0
        n_trees = len(picks)

        def frame_cb(i: int, n: int, phase: str, tree_i: int) -> None:
            if i == 1 or i == n or i % 9 == 0:
                print(
                    f"  tree {tree_i + 1}/{n_trees} {phase} frame {i}/{n}",
                    flush=True,
                )

        for i, pick in enumerate(picks):
            print(
                f"\n=== tree {i + 1}/{n_trees} {pick.chip_id} "
                f"bev={pick.bev_score:.2f} ver={pick.verifier_score:.2f} ===",
                flush=True,
            )
            def _map_cb(fi: int, fn: int, ph: str, _ti: int = i) -> None:
                frame_cb(fi, fn, ph, _ti)

            map_frames_list = render_map_zoom_frames(
                pick,
                register_xy=reg,
                city_bounds=bounds,
                n_frames=map_frames,
                on_frame=_map_cb,
            )
            print(f"  map segment done ({len(map_frames_list)} frames)", flush=True)

            def _cloud_cb(fi: int, fn: int, ph: str, _ti: int = i) -> None:
                frame_cb(fi, fn, ph, _ti)

            cloud_frames_list = render_cloud_orbit_frames(
                pick,
                n_frames=cloud_frames,
                rotations=cloud_rotations,
                mode_blend=cloud_mode_blend,
                mode_cycles=cloud_mode_cycles,
                crop_seed=hash((pick.x, pick.y, i)) % (2**31),
                on_frame=_cloud_cb,
            )
            print(f"  cloud segment done ({len(cloud_frames_list)} frames)", flush=True)
            segment = crossfade_frames(map_frames_list, cloud_frames_list, blend_frames)
            segment = dip_to_black(segment, black_frames)
            start = frame_idx
            seg_len = len(segment)
            for im in segment:
                frame_idx += 1
                p = tmp_path / f"frame_{frame_idx:06d}.png"
                im.save(p, format="PNG")
            segment.clear()
            map_frames_list.clear()
            cloud_frames_list.clear()
            print(f"  saved through frame {frame_idx} (disk)", flush=True)

            manifest["picks"].append(
                {
                    **asdict(pick),
                    "frame_start": start,
                    "frame_end": frame_idx,
                    "quality": pick.quality,
                }
            )

        print(f"\nencoding MP4 ({frame_idx} frames @ {fps} fps)...", flush=True)
        encode_mp4(tmp_path, frame_idx, output_mp4, fps=fps)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_mp4} ({frame_idx} frames @ {fps} fps)", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


def main() -> None:
    default_verified = exports_dir() / "inference" / "live" / "verified_predictions.json"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verified", type=Path, default=default_verified)
    p.add_argument(
        "--output",
        type=Path,
        default=exports_dir() / "inference" / "linkedin_verifier_reel.mp4",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=exports_dir() / "inference" / "linkedin_verifier_reel_manifest.json",
    )
    p.add_argument("--trees", type=Path, default=manifests_dir() / "gent_trees.parquet")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--bev-min", type=float, default=0.85)
    p.add_argument("--verifier-min", type=float, default=0.85)
    p.add_argument("--min-spacing-m", type=float, default=400.0)
    p.add_argument("--map-frames", type=int, default=27)
    p.add_argument(
        "--cloud-frames",
        type=int,
        default=120,
        help="Point-cloud segment length per tree (map zoom unchanged)",
    )
    p.add_argument(
        "--cloud-rotations",
        type=float,
        default=1.4,
        help="Full orbit turns across the cloud segment (eased)",
    )
    p.add_argument("--cloud-mode-blend", type=int, default=10)
    p.add_argument(
        "--cloud-mode-cycles",
        type=int,
        default=2,
        help="Alternations between RGB splat and height points (2 => 4 blocks)",
    )
    p.add_argument("--blend-frames", type=int, default=6)
    p.add_argument("--black-frames", type=int, default=4)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    def abs_path(path: Path) -> Path:
        return path if path.is_absolute() else repo_root() / path

    build_reel(
        verified_path=abs_path(args.verified),
        output_mp4=abs_path(args.output),
        manifest_path=abs_path(args.manifest),
        trees_parquet=abs_path(args.trees),
        n=args.n,
        bev_min=args.bev_min,
        verifier_min=args.verifier_min,
        min_spacing_m=args.min_spacing_m,
        map_frames=args.map_frames,
        cloud_frames=args.cloud_frames,
        cloud_rotations=args.cloud_rotations,
        cloud_mode_blend=args.cloud_mode_blend,
        cloud_mode_cycles=args.cloud_mode_cycles,
        blend_frames=args.blend_frames,
        black_frames=args.black_frames,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
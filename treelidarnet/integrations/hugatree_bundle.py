"""Build HugATree static LiDAR bundle (TLEX detections + zstd crop pack)."""

from __future__ import annotations

import mmap
import struct
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from treelidarnet.data.crops import (
    DEFAULT_CROP_RADIUS_M,
    DEFAULT_MIN_POINTS,
    DEFAULT_NUM_POINTS,
    IN_CHANNELS_RGB,
    chip_id_for_point,
    chip_origin_for_point,
)
from treelidarnet.data.crs import wgs84_to_lambert
from treelidarnet.explorer.bundle import (
    BUNDLE_MAGIC,
    BuildBundleConfig,
    _read_predictions,
    _row_scores,
    build_explorer_bundle,
    default_bundle_paths,
)
from treelidarnet.integrations.hugatree_crop_codec import CROP_CODEC_VERSION, encode_crop_tensor
from treelidarnet.paths import manifests_dir
from treelidarnet.postprocess import Detection
from treelidarnet.verifier_batch import _crop_at_detection, _load_chip_npz

IDX_MAGIC = b"HIDX"
IDX_VERSION = 1
IDX_ROW_STRUCT = struct.Struct("<ddddddQ I")  # lon,lat,bev,v2,v3,dist, off, len


@dataclass
class BuildHugatreeBundleConfig:
    input_path: Path
    output_dir: Path
    trees_parquet: Path
    radius_m: float = DEFAULT_CROP_RADIUS_M
    num_points: int = DEFAULT_NUM_POINTS
    min_points: int = DEFAULT_MIN_POINTS
    combo_threshold: float = 0.45
    combo_w_v2: float = 0.5
    combo_w_v3: float = 0.5
    limit: int | None = None


def _write_crop_index(rows: list[dict[str, Any]], idx_path: Path) -> None:
    n = len(rows)
    header = IDX_MAGIC + struct.pack("<II", IDX_VERSION, n)
    body = bytearray()
    for row in rows:
        body.extend(
            IDX_ROW_STRUCT.pack(
                float(row["lon"]),
                float(row["lat"]),
                float(row["bev"]),
                float(row["v2"]),
                float(row["v3"]),
                float(row["dist"]),
                int(row["offset"]),
                int(row["len"]),
            )
        )
    idx_path.write_bytes(header + bytes(body))


def build_hugatree_bundle(cfg: BuildHugatreeBundleConfig) -> dict[str, Any]:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    pred_rows = _read_predictions(cfg.input_path)
    if cfg.limit is not None:
        pred_rows = pred_rows[: cfg.limit]

    bundle_input = cfg.input_path
    subset_path = out / ".subset_predictions.json"
    if cfg.limit is not None:
        subset_path.write_bytes(orjson.dumps({"predictions": pred_rows}))
        bundle_input = subset_path

    bundle_cfg = BuildBundleConfig(
        input_path=bundle_input,
        output_dir=out,
        trees_parquet=cfg.trees_parquet,
        mark_rgb=False,
    )
    det_meta = build_explorer_bundle(bundle_cfg)
    reg_path = out / "register.bin"
    if reg_path.is_file():
        reg_path.unlink()
    if subset_path.is_file():
        subset_path.unlink()

    from treelidarnet.explorer.bundle import DETECTION_DTYPE

    header_size = len(BUNDLE_MAGIC) + 8
    det_arr = np.frombuffer(
        (out / "detections.bin").read_bytes()[header_size:],
        dtype=DETECTION_DTYPE,
    )

    detections: list[Detection] = []
    row_meta: list[dict[str, float]] = []
    for row in pred_rows:
        pos = row.get("position") or [0.0, 0.0]
        lon, lat = float(pos[0]), float(pos[1])
        bev, v2, v3 = _row_scores(row)
        x, y = wgs84_to_lambert(lon, lat)
        detections.append(Detection(x=x, y=y, score=bev))
        row_meta.append({"lon": lon, "lat": lat, "bev": bev, "v2": v2, "v3": v3})

    pack_path = out / "crops.pack.zst"
    idx_rows: list[dict[str, Any]] = []
    offset = 0
    pack_total = 0

    by_chip: dict[str, list[int]] = defaultdict(list)
    for i, det in enumerate(detections):
        by_chip[chip_id_for_point(det.x, det.y)].append(i)

    with pack_path.open("wb") as pack_f:
        for cid in sorted(by_chip.keys()):
            local_ix = by_chip[cid]
            chip = _load_chip_npz(cid, need_rgb=True)
            det0 = detections[local_ix[0]]
            origin_x, origin_y = chip_origin_for_point(det0.x, det0.y)
            for local_i in local_ix:
                det = detections[local_i]
                meta = row_meta[local_i]
                blob = b""
                if chip is not None:
                    xyz, feat, rgb = chip
                    arr = _crop_at_detection(
                        det=det,
                        origin_x=origin_x,
                        origin_y=origin_y,
                        xyz=xyz,
                        feat=feat,
                        rgb=rgb,
                        global_index=local_i,
                        channel_variant=1,
                        radius_m=cfg.radius_m,
                        num_points=cfg.num_points,
                        min_points=cfg.min_points,
                        in_channels=IN_CHANNELS_RGB,
                    )
                    if arr is not None:
                        blob = encode_crop_tensor(arr)
                pack_f.write(blob)
                idx_rows.append(
                    {
                        "lon": meta["lon"],
                        "lat": meta["lat"],
                        "bev": meta["bev"],
                        "v2": meta["v2"],
                        "v3": meta["v3"],
                        "dist": float(det_arr["dist_register_m"][local_i]),
                        "offset": offset,
                        "len": len(blob),
                    }
                )
                offset += len(blob)
                pack_total += len(blob)

    _write_crop_index(idx_rows, out / "crops.idx")

    meta = {
        "detection_count": int(det_meta["detection_count"]),
        "export_path": str(cfg.input_path.resolve()),
        "built_at": datetime.now(UTC).isoformat(),
        "crop_format": {"codec_version": CROP_CODEC_VERSION, "zstd_level": 19, "quant_mm": True},
        "total_pack_bytes": int(pack_total),
        "combo_threshold": cfg.combo_threshold,
        "combo_w_v2": cfg.combo_w_v2,
        "combo_w_v3": cfg.combo_w_v3,
        "verifier_w_v2": cfg.combo_w_v2,
        "verifier_w_v3": cfg.combo_w_v3,
        "crop_count": len(idx_rows),
        "radius_m": cfg.radius_m,
        "num_points": cfg.num_points,
    }
    (out / "meta.json").write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    return meta


def default_hugatree_output() -> Path:
    return Path("/home/catt/Dev/HugATree/static/data/lidar")


def cli_build_hugatree_bundle(
    *,
    input_path: Path | None,
    output_dir: Path | None,
    trees_parquet: Path | None,
    limit: int | None,
    combo_threshold: float,
) -> None:
    default_in, _ = default_bundle_paths()
    cfg = BuildHugatreeBundleConfig(
        input_path=input_path or default_in,
        output_dir=output_dir or default_hugatree_output(),
        trees_parquet=trees_parquet or (manifests_dir() / "gent_trees.parquet"),
        limit=limit,
        combo_threshold=combo_threshold,
    )
    meta = build_hugatree_bundle(cfg)
    print(
        f"HugATree bundle: {meta['detection_count']} detections, "
        f"pack {meta['total_pack_bytes'] / 1e6:.1f} MB → {cfg.output_dir}"
    )


def load_idx_mmap(idx_path: Path) -> tuple[mmap.mmap, int]:
    f = idx_path.open("rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if mm[:4] != IDX_MAGIC:
        raise ValueError("bad crops.idx magic")
    version, count = struct.unpack_from("<II", mm, 4)
    if version != IDX_VERSION:
        raise ValueError(f"unsupported crops.idx version {version}")
    return mm, count
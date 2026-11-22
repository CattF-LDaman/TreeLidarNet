"""Quantized LiDAR crop blobs for HugATree (zstd frames in crops.pack.zst)."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
import zstandard as zstd

CROP_CODEC_VERSION = 1
_ZSTD_LEVEL = 19


def encode_crop_tensor(arr: np.ndarray) -> bytes:
    """Pack local xyz (m), intensity 0–1, rgb 0–1 into a zstd frame."""
    n = int(arr.shape[0])
    buf = bytearray()
    buf.extend(struct.pack("<H", n))
    imax = float(np.max(arr[:, 3])) if n else 0.0
    if imax <= 0.0:
        imax = 1.0
    for row in arr:
        x_mm = int(np.clip(round(float(row[0]) * 1000.0), -32768, 32767))
        y_mm = int(np.clip(round(float(row[1]) * 1000.0), -32768, 32767))
        z_mm = int(np.clip(round(float(row[2]) * 1000.0), -32768, 32767))
        rel_i = float(row[3]) / imax
        intensity = int(np.clip(round(rel_i * 255.0), 0, 255))
        if row.shape[0] >= 9:
            r = int(np.clip(round(float(row[6]) * 255.0), 0, 255))
            g = int(np.clip(round(float(row[7]) * 255.0), 0, 255))
            b = int(np.clip(round(float(row[8]) * 255.0), 0, 255))
        else:
            r = g = b = 0
        buf.extend(struct.pack("<hhhBBBB", x_mm, y_mm, z_mm, intensity, r, g, b))
    return zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress(bytes(buf))


def decode_crop_blob(blob: bytes) -> list[dict[str, float]]:
    raw = zstd.ZstdDecompressor().decompress(blob)
    if len(raw) < 2:
        return []
    (n,) = struct.unpack_from("<H", raw, 0)
    off = 2
    out: list[dict[str, float]] = []
    for _ in range(n):
        x_mm, y_mm, z_mm, intensity_u8, r, g, b = struct.unpack_from("<hhhBBBB", raw, off)
        off += 10
        out.append(
            {
                "x": x_mm / 1000.0,
                "y": y_mm / 1000.0,
                "z": z_mm / 1000.0,
                # Intensity uint8 is max-normalized within each packed crop (0 = min, 1 = max).
                "intensity": intensity_u8 / 255.0,
                "red": r / 255.0,
                "green": g / 255.0,
                "blue": b / 255.0,
            }
        )
    return out


def crop_json_from_points(
    *,
    points: list[dict[str, float]],
    lon: float,
    lat: float,
    bev: float,
    v2: float,
    v3: float,
    dist_register_m: float,
    radius_m: float,
    combo_w_v2: float = 0.5,
    combo_w_v3: float = 0.5,
) -> dict[str, Any]:
    combo = combo_w_v2 * v2 + combo_w_v3 * v3
    return {
        "position": [lon, lat],
        "num_points": len(points),
        "radius_m": radius_m,
        "points": points,
        "score": bev,
        "verifier_scores": {"v2": v2, "v3": v3},
        "combo": combo,
        "dist_register_m": dist_register_m,
    }
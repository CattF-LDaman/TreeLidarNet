"""Run crop verifier on Lambert detections."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from treelidarnet.data.crops import extract_crop_at_global
from treelidarnet.models.crop_verifier import GentCropVerifier, load_crop_verifier
from treelidarnet.postprocess import Detection


def score_detection(
    model: GentCropVerifier,
    *,
    x: float,
    y: float,
    device: torch.device,
    radius_m: float = 8.0,
    num_points: int = 512,
    min_points: int = 32,
    seed: int | None = None,
) -> float:
    """Tree probability for one global Lambert point."""
    rng = np.random.default_rng(seed) if seed is not None else None
    crop = extract_crop_at_global(
        global_x=x,
        global_y=y,
        radius_m=radius_m,
        num_points=num_points,
        min_points=min_points,
        rng=rng,
        in_channels=int(getattr(model, "in_channels", 6)),
    )
    if crop is None:
        return 0.0
    pts = torch.from_numpy(crop).unsqueeze(0).to(device)
    with torch.inference_mode():
        logit = model(pts)
        return float(torch.sigmoid(logit).item())


def score_detections_with_model(
    detections: list[Detection],
    model: GentCropVerifier,
    *,
    device: torch.device,
    radius_m: float = 8.0,
    num_points: int = 512,
    min_points: int = 32,
    progress_base: int = 0,
) -> list[Detection]:
    out: list[Detection] = []
    for i, det in enumerate(detections):
        prob = score_detection(
            model,
            x=det.x,
            y=det.y,
            device=device,
            radius_m=radius_m,
            num_points=num_points,
            min_points=min_points,
            seed=hash((det.x, det.y, progress_base + i)) % (2**31),
        )
        out.append(replace(det, verifier_score=prob))
    return out


def score_detections_dual(
    detections: list[Detection],
    model_v2: GentCropVerifier,
    model_v3: GentCropVerifier,
    *,
    device: torch.device,
    radius_m: float = 8.0,
    num_points: int = 512,
    min_points: int = 32,
    progress_base: int = 0,
) -> tuple[list[Detection], list[dict[str, float]]]:
    """Score each detection with v2 (6ch) and v3 (9ch+RGB); same points, same order."""
    scored_v2 = score_detections_with_model(
        detections,
        model_v2,
        device=device,
        radius_m=radius_m,
        num_points=num_points,
        min_points=min_points,
        progress_base=progress_base,
    )
    scored_v3 = score_detections_with_model(
        detections,
        model_v3,
        device=device,
        radius_m=radius_m,
        num_points=num_points,
        min_points=min_points,
        progress_base=progress_base + 1,
    )
    out: list[Detection] = []
    scores: list[dict[str, float]] = []
    for det, d2, d3 in zip(detections, scored_v2, scored_v3, strict=True):
        v2 = float(d2.verifier_score or 0.0)
        v3 = float(d3.verifier_score or 0.0)
        scores.append({"v2": v2, "v3": v3})
        out.append(replace(det, verifier_score=v3))
    return out, scores


def score_detections(
    detections: list[Detection],
    *,
    checkpoint: Path,
    device: torch.device | None = None,
    radius_m: float = 8.0,
    num_points: int = 512,
    min_points: int = 32,
) -> list[Detection]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_crop_verifier(checkpoint, device)
    return score_detections_with_model(
        detections,
        model,
        device=device,
        radius_m=radius_m,
        num_points=num_points,
        min_points=min_points,
    )
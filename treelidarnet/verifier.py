"""Run crop verifier on Lambert detections."""

from __future__ import annotations

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
    infer_batch_size: int = 64,
    on_progress=None,
) -> list[Detection]:
    from treelidarnet.verifier_batch import score_detections_batched

    return score_detections_batched(
        detections,
        model,
        device=device,
        radius_m=radius_m,
        num_points=num_points,
        min_points=min_points,
        progress_base=progress_base,
        infer_batch_size=infer_batch_size,
        on_progress=on_progress,
    )


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
    infer_batch_size: int = 64,
    on_progress=None,
) -> tuple[list[Detection], list[dict[str, float]]]:
    """Score each detection with v2 (6ch) and v3 (9ch+RGB); same order as legacy loop."""
    from treelidarnet.verifier_batch import score_detections_dual_batched

    return score_detections_dual_batched(
        detections,
        model_v2,
        model_v3,
        device=device,
        radius_m=radius_m,
        num_points=num_points,
        min_points=min_points,
        progress_base=progress_base,
        infer_batch_size=infer_batch_size,
        on_progress=on_progress,
    )


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
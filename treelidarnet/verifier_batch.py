"""Batched crop-verifier scoring (chip grouping + GPU batch forward)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import torch

from treelidarnet.data.crops import (
    IN_CHANNELS,
    IN_CHANNELS_RGB,
    build_crop_tensor,
    chip_id_for_point,
    chip_origin_for_point,
    global_to_chip_local,
)
from treelidarnet.models.crop_verifier import GentCropVerifier
from treelidarnet.paths import chips_dir
from treelidarnet.postprocess import Detection


def _det_seed(x: float, y: float, global_index: int) -> int:
    return hash((x, y, global_index)) % (2**31)


def _load_chip_npz(
    chip_id: str, *, need_rgb: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    path = chips_dir() / f"{chip_id}.npz"
    if not path.is_file():
        return None
    data = np.load(path)
    try:
        xyz = data["xyz"].astype(np.float32)
        feat = data["feat"].astype(np.float32)
        rgb = None
        if need_rgb and "rgb" in data.files:
            rgb = data["rgb"].astype(np.float32)
        return xyz, feat, rgb
    finally:
        data.close()


def _crop_at_detection(
    *,
    det: Detection,
    origin_x: float,
    origin_y: float,
    xyz: np.ndarray,
    feat: np.ndarray,
    rgb: np.ndarray | None,
    global_index: int,
    channel_variant: int,
    radius_m: float,
    num_points: int,
    min_points: int,
    in_channels: int,
) -> np.ndarray | None:
    """Match legacy seeds: v2 variant 0 → hash(x,y,gi); v3 variant 1 → hash(x,y,gi+1)."""
    local_x, local_y = global_to_chip_local(det.x, det.y, origin_x=origin_x, origin_y=origin_y)
    seed = _det_seed(det.x, det.y, global_index + channel_variant)
    rng = np.random.default_rng(seed)
    return build_crop_tensor(
        xyz,
        feat,
        center_x=local_x,
        center_y=local_y,
        radius_m=radius_m,
        num_points=num_points,
        min_points=min_points,
        rng=rng,
        rgb=rgb,
        in_channels=in_channels,
    )


def _forward_probs(
    model: GentCropVerifier,
    crops: list[np.ndarray],
    device: torch.device,
    infer_batch_size: int,
) -> list[float]:
    if not crops:
        return []
    out: list[float] = []
    for start in range(0, len(crops), infer_batch_size):
        chunk = crops[start : start + infer_batch_size]
        batch = torch.from_numpy(np.stack(chunk, axis=0)).to(device)
        with torch.inference_mode():
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy()
        if probs.ndim == 0:
            out.append(float(probs))
        else:
            out.extend(float(p) for p in probs)
    return out


def _flush_dual_batch(
    *,
    model_v2: GentCropVerifier,
    model_v3: GentCropVerifier,
    pending_idx: list[int],
    pending_v2: list[np.ndarray],
    pending_v3: list[np.ndarray],
    v2_out: list[float],
    v3_out: list[float],
    device: torch.device,
    infer_batch_size: int,
) -> None:
    if not pending_idx:
        return
    p2 = _forward_probs(model_v2, pending_v2, device, infer_batch_size)
    p3 = _forward_probs(model_v3, pending_v3, device, infer_batch_size)
    for li, a, b in zip(pending_idx, p2, p3, strict=True):
        v2_out[li] = a
        v3_out[li] = b
    pending_idx.clear()
    pending_v2.clear()
    pending_v3.clear()


def _flush_single_batch(
    *,
    model: GentCropVerifier,
    pending_idx: list[int],
    pending: list[np.ndarray],
    scores: list[float],
    device: torch.device,
    infer_batch_size: int,
) -> None:
    if not pending_idx:
        return
    ps = _forward_probs(model, pending, device, infer_batch_size)
    for li, p in zip(pending_idx, ps, strict=True):
        scores[li] = p
    pending_idx.clear()
    pending.clear()


def score_detections_dual_batched(
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
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[Detection], list[dict[str, float]]]:
    """Same scores/order as legacy dual path; groups by chip and batches GPU forward."""
    n = len(detections)
    v2_scores = [0.0] * n
    v3_scores = [0.0] * n

    by_chip: dict[str, list[int]] = defaultdict(list)
    for local_i, det in enumerate(detections):
        by_chip[chip_id_for_point(det.x, det.y)].append(local_i)

    pending_idx: list[int] = []
    pending_v2: list[np.ndarray] = []
    pending_v3: list[np.ndarray] = []
    done = 0

    for cid in sorted(by_chip.keys()):
        local_indices = by_chip[cid]
        chip = _load_chip_npz(cid, need_rgb=True)
        if chip is None:
            done += len(local_indices)
            if on_progress:
                on_progress(done, n)
            continue
        xyz, feat, rgb = chip
        det0 = detections[local_indices[0]]
        origin_x, origin_y = chip_origin_for_point(det0.x, det0.y)

        for local_i in local_indices:
            det = detections[local_i]
            gi = progress_base + local_i
            c2 = _crop_at_detection(
                det=det,
                origin_x=origin_x,
                origin_y=origin_y,
                xyz=xyz,
                feat=feat,
                rgb=rgb,
                global_index=gi,
                channel_variant=0,
                radius_m=radius_m,
                num_points=num_points,
                min_points=min_points,
                in_channels=IN_CHANNELS,
            )
            c3 = _crop_at_detection(
                det=det,
                origin_x=origin_x,
                origin_y=origin_y,
                xyz=xyz,
                feat=feat,
                rgb=rgb,
                global_index=gi,
                channel_variant=1,
                radius_m=radius_m,
                num_points=num_points,
                min_points=min_points,
                in_channels=IN_CHANNELS_RGB,
            )
            if c2 is not None and c3 is not None:
                pending_idx.append(local_i)
                pending_v2.append(c2)
                pending_v3.append(c3)
                if len(pending_idx) >= infer_batch_size:
                    _flush_dual_batch(
                        model_v2=model_v2,
                        model_v3=model_v3,
                        pending_idx=pending_idx,
                        pending_v2=pending_v2,
                        pending_v3=pending_v3,
                        v2_out=v2_scores,
                        v3_out=v3_scores,
                        device=device,
                        infer_batch_size=infer_batch_size,
                    )
            done += 1
            if on_progress and done % 512 == 0:
                on_progress(done, n)

        _flush_dual_batch(
            model_v2=model_v2,
            model_v3=model_v3,
            pending_idx=pending_idx,
            pending_v2=pending_v2,
            pending_v3=pending_v3,
            v2_out=v2_scores,
            v3_out=v3_scores,
            device=device,
            infer_batch_size=infer_batch_size,
        )

    if on_progress:
        on_progress(n, n)

    out_dets: list[Detection] = []
    scores: list[dict[str, float]] = []
    for local_i, det in enumerate(detections):
        v2 = v2_scores[local_i]
        v3 = v3_scores[local_i]
        scores.append({"v2": v2, "v3": v3})
        out_dets.append(replace(det, verifier_score=v3))
    return out_dets, scores


def score_detections_batched(
    detections: list[Detection],
    model: GentCropVerifier,
    *,
    device: torch.device,
    radius_m: float = 8.0,
    num_points: int = 512,
    min_points: int = 32,
    progress_base: int = 0,
    infer_batch_size: int = 64,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Detection]:
    n = len(detections)
    in_ch = int(getattr(model, "in_channels", IN_CHANNELS))
    need_rgb = in_ch > IN_CHANNELS
    single_scores = [0.0] * n

    by_chip: dict[str, list[int]] = defaultdict(list)
    for local_i, det in enumerate(detections):
        by_chip[chip_id_for_point(det.x, det.y)].append(local_i)

    pending_idx: list[int] = []
    pending: list[np.ndarray] = []
    done = 0

    for cid in sorted(by_chip.keys()):
        local_indices = by_chip[cid]
        chip = _load_chip_npz(cid, need_rgb=need_rgb)
        if chip is None:
            done += len(local_indices)
            if on_progress:
                on_progress(done, n)
            continue
        xyz, feat, rgb = chip
        det0 = detections[local_indices[0]]
        origin_x, origin_y = chip_origin_for_point(det0.x, det0.y)

        for local_i in local_indices:
            det = detections[local_i]
            gi = progress_base + local_i
            crop = _crop_at_detection(
                det=det,
                origin_x=origin_x,
                origin_y=origin_y,
                xyz=xyz,
                feat=feat,
                rgb=rgb,
                global_index=gi,
                channel_variant=0,
                radius_m=radius_m,
                num_points=num_points,
                min_points=min_points,
                in_channels=in_ch,
            )
            if crop is not None:
                pending_idx.append(local_i)
                pending.append(crop)
                if len(pending_idx) >= infer_batch_size:
                    _flush_single_batch(
                        model=model,
                        pending_idx=pending_idx,
                        pending=pending,
                        scores=single_scores,
                        device=device,
                        infer_batch_size=infer_batch_size,
                    )
            done += 1
            if on_progress and done % 512 == 0:
                on_progress(done, n)

        _flush_single_batch(
            model=model,
            pending_idx=pending_idx,
            pending=pending,
            scores=single_scores,
            device=device,
            infer_batch_size=infer_batch_size,
        )

    if on_progress:
        on_progress(n, n)

    return [replace(det, verifier_score=single_scores[i]) for i, det in enumerate(detections)]
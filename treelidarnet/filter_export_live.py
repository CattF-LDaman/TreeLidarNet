"""Live verifier scoring for Deck.GL polling (crop QA viewer)."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import orjson
import torch

from treelidarnet.explorer.combine import export_keep_passes, weighted_combo
from treelidarnet.filter_export_resume import load_scoring_resume
from treelidarnet.live_export import detections_to_rows, write_live_predictions, write_status
from treelidarnet.models.crop_verifier import load_crop_verifier
from treelidarnet.paths import exports_dir, repo_root
from treelidarnet.postprocess import Detection
from treelidarnet.verifier import score_detections_dual, score_detections_with_model
from treelidarnet.verify_export import (
    FilterExportResult,
    _load_predictions,
    register_distances_m,
)


def _resolve_live_inputs(
    *,
    in_path: Path,
    out_path: Path,
    trees_parquet: Path,
    bev_min: float,
    limit: int | None,
    require_chip_rgb: bool,
    live_dir: Path | None,
) -> tuple[Path, Path, list[Detection], Path, Path]:
    if not in_path.is_absolute():
        in_path = repo_root() / in_path
    if not out_path.is_absolute():
        out_path = repo_root() / out_path
    live_dir = live_dir or (exports_dir() / "inference" / "live")
    live_dir.mkdir(parents=True, exist_ok=True)
    pred_path = live_dir / "verified_predictions.json"
    status_path = live_dir / "verified_status.json"

    preds = _load_predictions(in_path)
    if bev_min > 0:
        preds = [p for p in preds if p.score >= bev_min]
    if limit is not None:
        preds = preds[:limit]

    if require_chip_rgb:
        from treelidarnet.data.rgb_chip_gate import filter_detections_on_rgb_chips

        preds, rgb_stats = filter_detections_on_rgb_chips(preds, run_id="live")
        print(
            f"  require-chip-rgb: scoring {rgb_stats['rgb_kept']}/{rgb_stats['input']} "
            f"(dropped no_rgb={rgb_stats['dropped_no_rgb']})",
            flush=True,
        )

    _ = trees_parquet  # loaded by caller
    return in_path, out_path, preds, pred_path, status_path


def _filter_kept_predictions(
    *,
    scored_all: list[Detection],
    dual_scores: list[dict[str, float]] | None,
    gts,
    compare: bool,
    combine_mode: str,
    verifier_threshold: float,
    w_v2: float,
    w_v3: float,
    discovery_m: float,
    register_m: float,
) -> tuple[list[Detection], list[dict[str, float]] | None, FilterExportResult]:
    xs = np.array([p.x for p in scored_all], dtype=np.float64)
    ys = np.array([p.y for p in scored_all], dtype=np.float64)
    dists = register_distances_m(gts, xs, ys)
    print(f"  register KD-tree distances for {len(dists)} detections", flush=True)

    discovery_in = int((dists > discovery_m).sum())
    reg_in = int((dists <= register_m).sum())

    kept: list[Detection] = []
    kept_scores: list[dict[str, float]] | None = [] if compare else None
    discovery_kept = reg_kept = 0
    for i, p in enumerate(scored_all):
        d = float(dists[i])
        if compare and dual_scores is not None:
            v2s = float(dual_scores[i].get("v2", 0.0))
            v3s = float(dual_scores[i].get("v3", 0.0))
            if not export_keep_passes(
                v2=v2s,
                v3=v3s,
                combine_mode=combine_mode,
                threshold=verifier_threshold,
                w_v2=w_v2,
                w_v3=w_v3,
            ):
                continue
        else:
            vs = p.verifier_score if p.verifier_score is not None else 0.0
            if vs < verifier_threshold:
                continue
        kept.append(p)
        if kept_scores is not None and dual_scores is not None:
            row_scores = dict(dual_scores[i])
            row_scores["combo"] = weighted_combo(
                float(row_scores.get("v2", 0.0)),
                float(row_scores.get("v3", 0.0)),
                w_v2=w_v2,
                w_v3=w_v3,
            )
            kept_scores.append(row_scores)
        if d > discovery_m:
            discovery_kept += 1
        if d <= register_m:
            reg_kept += 1

    result = FilterExportResult(
        input_count=len(scored_all),
        kept_count=len(kept),
        discovery_input=discovery_in,
        discovery_kept=discovery_kept,
        discovery_retention=discovery_kept / discovery_in if discovery_in else 1.0,
        register_near_input=reg_in,
        register_near_kept=reg_kept,
        register_retention=reg_kept / reg_in if reg_in else 1.0,
    )
    return kept, kept_scores, result


def finalize_filter_export_live(
    *,
    in_path: Path,
    out_path: Path,
    trees_parquet: Path,
    live_pred_path: Path,
    status_path: Path,
    verifier_checkpoint: Path,
    verifier_threshold: float = 0.5,
    bev_min: float = 0.0,
    discovery_m: float = 5.0,
    register_m: float = 3.0,
    verifier_checkpoint_v2: Path | None = None,
    verifier_checkpoint_v3: Path | None = None,
    combine_mode: str = "weighted",
    w_v2: float = 0.5,
    w_v3: float = 0.5,
) -> FilterExportResult:
    """Write verified gzip from existing live JSON (skip GPU scoring)."""
    from treelidarnet.eval import load_ground_truths

    compare = verifier_checkpoint_v2 is not None
    v3_ckpt = verifier_checkpoint_v3 or verifier_checkpoint
    if not v3_ckpt.is_absolute():
        v3_ckpt = repo_root() / v3_ckpt

    if not in_path.is_absolute():
        in_path = repo_root() / in_path
    if not out_path.is_absolute():
        out_path = repo_root() / out_path

    preds = _load_predictions(in_path)
    if bev_min > 0:
        preds = [p for p in preds if p.score >= bev_min]

    payload = orjson.loads(live_pred_path.read_bytes())
    rows = payload.get("predictions") or []
    if len(rows) != len(preds):
        raise ValueError(
            f"live predictions {len(rows)} != scored pool {len(preds)} "
            f"(bev_min={bev_min}); cannot finalize"
        )

    dual_scores: list[dict[str, float]] | None = [] if compare else None
    scored_all: list[Detection] = []
    for p, row in zip(preds, rows, strict=True):
        vs = row.get("verifier_scores")
        if compare and isinstance(vs, dict):
            v2 = float(vs.get("v2", 0.0))
            v3 = float(vs.get("v3", 0.0))
            dual_scores.append({"v2": v2, "v3": v3})
            scored_all.append(replace(p, verifier_score=v3))
        else:
            v = float(row.get("verifier_score") or 0.0)
            scored_all.append(replace(p, verifier_score=v))

    print(
        f"finalize filter-export: keep phase from {live_pred_path} ({len(rows)} scored)",
        flush=True,
    )
    gts = load_ground_truths(trees_parquet, eligible_only=True)
    kept, kept_scores, result = _filter_kept_predictions(
        scored_all=scored_all,
        dual_scores=dual_scores,
        gts=gts,
        compare=compare,
        combine_mode=combine_mode,
        verifier_threshold=verifier_threshold,
        w_v2=w_v2,
        w_v3=w_v3,
        discovery_m=discovery_m,
        register_m=register_m,
    )

    with gzip.open(in_path, "rb") as handle:
        out_payload = json.loads(handle.read())
    pred_rows = detections_to_rows(kept, verifier_scores_list=kept_scores)
    out_payload["predictions"] = pred_rows
    out_payload["verifier_checkpoint"] = str(v3_ckpt)
    out_payload["verifier_threshold"] = verifier_threshold
    if compare and verifier_checkpoint_v2 is not None:
        v2p = verifier_checkpoint_v2
        if not v2p.is_absolute():
            v2p = repo_root() / v2p
        out_payload["verifier_checkpoint_v2"] = str(v2p)
        out_payload["compare_verifiers"] = True
        out_payload["verifier_combine_mode"] = combine_mode
        out_payload["verifier_w_v2"] = w_v2
        out_payload["verifier_w_v3"] = w_v3
        if combine_mode == "weighted":
            out_payload["verifier_combo_threshold"] = verifier_threshold
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as handle:
        json.dump(out_payload, handle)

    write_status(
        status_path,
        phase="done",
        done=len(preds),
        total=len(preds),
        scored=len(scored_all),
        kept=len(kept),
        bev_min=bev_min,
        verifier_threshold=verifier_threshold,
        output=str(out_path),
        compare=bool(compare),
        combine_mode=combine_mode if compare else None,
        w_v2=w_v2 if compare else None,
        w_v3=w_v3 if compare else None,
        finalized_from_live=str(live_pred_path),
    )
    print(
        f"filter-export: {result.kept_count}/{result.input_count} kept; "
        f"discovery retention {result.discovery_kept}/{result.discovery_input} "
        f"({100 * result.discovery_retention:.1f}%); "
        f"register @3m {result.register_near_kept}/{result.register_near_input} "
        f"({100 * result.register_retention:.1f}%)",
        flush=True,
    )
    return result


def filter_export_live(
    *,
    in_path: Path,
    out_path: Path,
    verifier_checkpoint: Path,
    trees_parquet: Path,
    verifier_threshold: float = 0.5,
    bev_min: float = 0.0,
    discovery_m: float = 5.0,
    register_m: float = 3.0,
    radius_m: float = 8.0,
    limit: int | None = None,
    batch_size: int = 48,
    infer_batch_size: int = 64,
    live_flush_every: int = 512,
    resume: bool = True,
    live_dir: Path | None = None,
    require_chip_rgb: bool = False,
    verifier_checkpoint_v2: Path | None = None,
    verifier_checkpoint_v3: Path | None = None,
    combine_mode: str = "weighted",
    w_v2: float = 0.5,
    w_v3: float = 0.5,
) -> FilterExportResult:
    """Score every input detection; stream partial rows to live_dir for the map."""
    compare = verifier_checkpoint_v2 is not None and (
        verifier_checkpoint_v3 is not None or verifier_checkpoint is not None
    )
    v3_ckpt = verifier_checkpoint_v3 or verifier_checkpoint
    if not v3_ckpt.is_absolute():
        v3_ckpt = repo_root() / v3_ckpt
    if not verifier_checkpoint.is_absolute():
        verifier_checkpoint = repo_root() / verifier_checkpoint

    in_path, out_path, preds, pred_path, status_path = _resolve_live_inputs(
        in_path=in_path,
        out_path=out_path,
        trees_parquet=trees_parquet,
        bev_min=bev_min,
        limit=limit,
        require_chip_rgb=require_chip_rgb,
        live_dir=live_dir,
    )

    from treelidarnet.eval import load_ground_truths

    gts = load_ground_truths(trees_parquet, eligible_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_v2 = model_v3 = None
    if compare:
        v2_ckpt = verifier_checkpoint_v2
        if not v2_ckpt.is_absolute():
            v2_ckpt = repo_root() / v2_ckpt
        model_v2 = load_crop_verifier(v2_ckpt, device)
        model_v3 = load_crop_verifier(v3_ckpt, device)
        gate = (
            f"combine={combine_mode} t={verifier_threshold}"
            if combine_mode == "weighted"
            else f"combine={combine_mode} threshold={verifier_threshold}"
        )
        if combine_mode == "weighted":
            gate += f" w=({w_v2},{w_v3})"
        print(
            f"filter-export --live compare: v2={v2_ckpt.name} v3={v3_ckpt.name} "
            f"on {len(preds)} detections (bev>={bev_min}; {gate})",
            flush=True,
        )
        status_gate = {
            "compare": True,
            "verifier_checkpoints": {"v2": str(v2_ckpt), "v3": str(v3_ckpt)},
            "combine_mode": combine_mode,
            "w_v2": w_v2,
            "w_v3": w_v3,
        }
        write_status(
            status_path,
            phase="scoring",
            done=0,
            total=len(preds),
            bev_min=bev_min,
            verifier_threshold=verifier_threshold,
            input=str(in_path),
            **status_gate,
        )
    else:
        model_v3 = load_crop_verifier(verifier_checkpoint, device)
        from treelidarnet.data.rgb_chip_gate import _agent_log

        _agent_log(
            hypothesis_id="C",
            location="filter_export_live.py:model",
            message="verifier loaded",
            data={
                "in_channels": int(getattr(model_v3, "in_channels", 0)),
                "require_chip_rgb": require_chip_rgb,
                "n_preds": len(preds),
                "runId": "live",
            },
        )
        print(
            f"filter-export --live: scoring {len(preds)} detections (bev>={bev_min})",
            flush=True,
        )
        write_status(
            status_path,
            phase="scoring",
            done=0,
            total=len(preds),
            bev_min=bev_min,
            verifier_threshold=verifier_threshold,
            verifier_checkpoint=str(verifier_checkpoint),
            input=str(in_path),
        )

    print(
        f"  live map: poll /api/live/verified (server) or {pred_path.parent} "
        f"(batched infer_batch_size={infer_batch_size})",
        flush=True,
    )

    resume_done = 0
    scored_all: list[Detection] = []
    dual_scores: list[dict[str, float]] | None = [] if compare else None
    if resume:
        resume_done, prefix_scored, prefix_dual = load_scoring_resume(
            status_path=status_path,
            live_pred_path=pred_path,
            preds=preds,
            compare=compare,
        )
        scored_all = prefix_scored
        if compare and dual_scores is not None and prefix_dual is not None:
            dual_scores = list(prefix_dual)

    remaining = preds[resume_done:]
    last_flush = resume_done

    def _maybe_flush(global_done: int) -> None:
        nonlocal last_flush
        if global_done - last_flush < live_flush_every and global_done < len(preds):
            return
        write_live_predictions(
            pred_path,
            scored_all,
            verifier_scores_list=dual_scores,
        )
        status_extra: dict = {}
        if compare:
            status_extra["compare"] = True
        write_status(
            status_path,
            phase="scoring",
            done=global_done,
            total=len(preds),
            scored=global_done,
            bev_min=bev_min,
            verifier_threshold=verifier_threshold,
            infer_batch_size=infer_batch_size,
            **status_extra,
        )
        print(f"  live flush {global_done}/{len(preds)}", flush=True)
        last_flush = global_done

    if remaining:
        if resume_done:
            print(f"  scoring remaining {len(remaining)} (resume from {resume_done})", flush=True)
        _ = batch_size  # legacy alias; scoring is no longer sliced by batch_size

        def on_progress(done_chunk: int, _chunk_total: int) -> None:
            _maybe_flush(resume_done + done_chunk)

        if compare and model_v2 is not None and model_v3 is not None and dual_scores is not None:
            scored_new, dual_new = score_detections_dual(
                remaining,
                model_v2,
                model_v3,
                device=device,
                radius_m=radius_m,
                progress_base=resume_done,
                infer_batch_size=infer_batch_size,
                on_progress=on_progress,
            )
            scored_all.extend(scored_new)
            dual_scores.extend(dual_new)
        else:
            assert model_v3 is not None
            scored_new = score_detections_with_model(
                remaining,
                model_v3,
                device=device,
                radius_m=radius_m,
                progress_base=resume_done,
                infer_batch_size=infer_batch_size,
                on_progress=on_progress,
            )
            scored_all.extend(scored_new)
        _maybe_flush(len(preds))
    elif resume_done and scored_all:
        print(f"  scoring complete from resume ({len(scored_all)} detections)", flush=True)

    kept, kept_scores, result = _filter_kept_predictions(
        scored_all=scored_all,
        dual_scores=dual_scores,
        gts=gts,
        compare=compare,
        combine_mode=combine_mode,
        verifier_threshold=verifier_threshold,
        w_v2=w_v2,
        w_v3=w_v3,
        discovery_m=discovery_m,
        register_m=register_m,
    )

    with gzip.open(in_path, "rb") as handle:
        payload = json.loads(handle.read())
    pred_rows = detections_to_rows(kept, verifier_scores_list=kept_scores)
    payload["predictions"] = pred_rows
    payload["verifier_checkpoint"] = str(v3_ckpt)
    payload["verifier_threshold"] = verifier_threshold
    if compare and verifier_checkpoint_v2 is not None:
        v2p = verifier_checkpoint_v2
        if not v2p.is_absolute():
            v2p = repo_root() / v2p
        payload["verifier_checkpoint_v2"] = str(v2p)
        payload["compare_verifiers"] = True
        payload["verifier_combine_mode"] = combine_mode
        payload["verifier_w_v2"] = w_v2
        payload["verifier_w_v3"] = w_v3
        if combine_mode == "weighted":
            payload["verifier_combo_threshold"] = verifier_threshold
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    write_live_predictions(
        pred_path,
        scored_all,
        verifier_scores_list=dual_scores,
    )
    write_status(
        status_path,
        phase="done",
        done=len(preds),
        total=len(preds),
        scored=len(scored_all),
        kept=len(kept),
        bev_min=bev_min,
        verifier_threshold=verifier_threshold,
        output=str(out_path),
        compare=bool(compare),
        combine_mode=combine_mode if compare else None,
        w_v2=w_v2 if compare else None,
        w_v3=w_v3 if compare else None,
    )

    print(
        f"filter-export: {result.kept_count}/{result.input_count} kept; "
        f"discovery retention {result.discovery_kept}/{result.discovery_input} "
        f"({100 * result.discovery_retention:.1f}%); "
        f"register @3m {result.register_near_kept}/{result.register_near_input} "
        f"({100 * result.register_retention:.1f}%)",
        flush=True,
    )
    return result
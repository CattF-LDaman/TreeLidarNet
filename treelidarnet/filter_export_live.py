"""Live verifier scoring for Deck.GL polling (crop QA viewer)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch

from treelidarnet.explorer.combine import export_keep_passes, weighted_combo
from treelidarnet.live_export import detections_to_rows, write_live_predictions, write_status
from treelidarnet.models.crop_verifier import load_crop_verifier
from treelidarnet.paths import exports_dir, repo_root
from treelidarnet.postprocess import Detection
from treelidarnet.verifier import score_detections_dual, score_detections_with_model
from treelidarnet.verify_export import FilterExportResult, _load_predictions, _nearest_gt_dist


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
    live_flush_every: int = 96,
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

    print(f"  live map: poll /api/live/verified (server) or {pred_path.parent}", flush=True)

    scored_all: list[Detection] = []
    dual_scores: list[dict[str, float]] | None = [] if compare else None
    last_flush = 0
    for start in range(0, len(preds), batch_size):
        batch = preds[start : start + batch_size]
        if compare and model_v2 is not None and model_v3 is not None and dual_scores is not None:
            scored, batch_scores = score_detections_dual(
                batch,
                model_v2,
                model_v3,
                device=device,
                radius_m=radius_m,
                progress_base=start,
            )
            scored_all.extend(scored)
            dual_scores.extend(batch_scores)
        else:
            assert model_v3 is not None
            scored = score_detections_with_model(
                batch,
                model_v3,
                device=device,
                radius_m=radius_m,
                progress_base=start,
            )
            scored_all.extend(scored)
        done = len(scored_all)
        if done - last_flush >= live_flush_every or done >= len(preds):
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
                done=done,
                total=len(preds),
                scored=done,
                bev_min=bev_min,
                verifier_threshold=verifier_threshold,
                **status_extra,
            )
            print(f"  live flush {done}/{len(preds)}", flush=True)
            last_flush = done

    discovery_in = reg_in = discovery_kept = reg_kept = 0
    kept: list[Detection] = []
    kept_scores: list[dict[str, float]] | None = [] if compare else None
    for i, p in enumerate(scored_all):
        d = _nearest_gt_dist(gts, p.x, p.y)
        if d > discovery_m:
            discovery_in += 1
        if d <= register_m:
            reg_in += 1
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

    result = FilterExportResult(
        input_count=len(preds),
        kept_count=len(kept),
        discovery_input=discovery_in,
        discovery_kept=discovery_kept,
        discovery_retention=discovery_kept / discovery_in if discovery_in else 1.0,
        register_near_input=reg_in,
        register_near_kept=reg_kept,
        register_retention=reg_kept / reg_in if reg_in else 1.0,
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
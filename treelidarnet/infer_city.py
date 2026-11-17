"""Run model on all Gent chips; export city-scale Deck.GL bundle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from treelidarnet.data.chips import CELL_SIZE_M, chip_id
from treelidarnet.datasets.chip_dataset import ChipDataset, collate_infer_chips
from treelidarnet.datasets.chip_index import chip_paths_outward_from_index, order_paths_outward
from treelidarnet.eval import load_ground_truths
from treelidarnet.export_mapjson import export_city_gz
from treelidarnet.infer import load_model
from treelidarnet.live_export import (
    export_gt_gz,
    read_live_predictions,
    read_live_status,
    write_live_predictions,
    write_status,
)
from treelidarnet.metrics import match_detections
from treelidarnet.paths import chips_dir, exports_dir
from treelidarnet.postprocess import (
    DEFAULT_PEAK_SCORE_THRESHOLD,
    Detection,
    extract_peaks_from_heatmap,
    merge_chip_detections,
    merge_detection_lists,
)


def _infer_worker(_worker_id: int) -> None:
    # One thread per worker. The scatter is a single sequential sum either way.
    torch.set_num_threads(1)


def infer_gent_city(
    *,
    checkpoint: Path,
    trees_parquet: Path,
    export_path: Path | None = None,
    model_id: str = "gent_canopy_v2",
    batch_size: int = 8,
    limit: int | None = None,
    chip_glob: str = "*.npz",
    chips_index: Path | None = None,
    live: bool = False,
    resume_live: bool = False,
    live_flush_every: int = 150,
    progress_every: int = 50,
    score_threshold: float = DEFAULT_PEAK_SCORE_THRESHOLD,
    num_workers: int = 4,
    verifier_checkpoint: Path | None = None,
    verifier_threshold: float | None = None,
    verifier_radius_m: float = 8.0,
) -> Path:
    if chips_index is not None and chips_index.is_file():
        paths, center = chip_paths_outward_from_index(chips_index)
        order_msg = "register chips first, then the rest; each outward"
    else:
        paths, center = order_paths_outward(sorted(chips_dir().glob(chip_glob)))
        order_msg = "outward"
    if not paths:
        raise FileNotFoundError(f"No chips in {chips_dir()}")
    if center is not None:
        print(
            f"infer order: {order_msg} from ({center[0]:.0f}, {center[1]:.0f})",
            flush=True,
        )
    total_paths = len(paths)
    if limit is not None:
        paths = paths[:limit]
        total_paths = len(paths)

    live_dir = exports_dir() / "inference" / "live" if live else None
    resume_done = 0
    committed_merged: list[Detection] = []
    if resume_live and live_dir is None:
        raise ValueError("--resume-live requires --live")

    if live_dir is not None:
        df = pd.read_parquet(trees_parquet)
        df = df[df["eligible"]]
        gt_path = live_dir / "gt.json.gz"
        if resume_live:
            status = read_live_status(live_dir / "status.json")
            pred_path = live_dir / "predictions.json"
            if status is None:
                raise FileNotFoundError(
                    f"--resume-live needs {live_dir / 'status.json'} from a previous --live run"
                )
            resume_done = int(status.get("done") or 0)
            if resume_done <= 0:
                raise ValueError("--resume-live: status done=0; run without --resume-live")
            if resume_done >= total_paths:
                raise ValueError(
                    f"--resume-live: already at {resume_done}/{total_paths} chips; nothing to do"
                )
            committed_merged = read_live_predictions(pred_path)
            paths = paths[resume_done:]
            print(
                f"resume live: skipping {resume_done} chips; "
                f"loaded {len(committed_merged)} preds from {pred_path.name}",
                flush=True,
            )
        else:
            export_gt_gz(gt_path, df.to_dict(orient="records"))
            write_status(
                live_dir / "status.json",
                state="running",
                done=0,
                total=total_paths,
                pred_count=0,
                model_id=model_id,
            )
        print(f"live map: poll viewer/ghent_live.html (data under {live_dir})", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"infer {len(paths)} chips"
        + (f" (resume from {resume_done}/{total_paths})" if resume_done else "")
        + f" batch={batch_size} device={device} workers={num_workers} "
        f"min_confidence={score_threshold}",
        flush=True,
    )
    model = load_model(checkpoint, device)
    ds = ChipDataset(paths, labels=False)
    use_cuda = device.type == "cuda"
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_infer_chips,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = _infer_worker
    loader = DataLoader(ds, **loader_kwargs)

    pending_chips: list[tuple[str, list[Detection]]] = []
    all_chips: list[tuple[str, list[Detection]]] = []
    idx = 0
    live_pred_count = len(committed_merged)
    last_live_flush = resume_done
    merge_radius_m = 2.5

    def _flush_live(done: int, *, final: bool = False) -> None:
        nonlocal committed_merged, live_pred_count, last_live_flush, pending_chips
        if live_dir is None or not pending_chips:
            return
        delta = merge_chip_detections(pending_chips, radius_m=merge_radius_m)
        pending_chips.clear()
        if committed_merged:
            committed_merged = merge_detection_lists(
                committed_merged, delta, radius_m=merge_radius_m
            )
        else:
            committed_merged = delta
        live_pred_count = len(committed_merged)
        last_live_flush = done
        write_live_predictions(live_dir / "predictions.json", committed_merged)
        write_status(
            live_dir / "status.json",
            state="merging" if final and done >= total_paths else "running",
            done=done,
            total=total_paths,
            pred_count=live_pred_count,
            model_id=model_id,
        )
        print(f"  live flush {done}/{total_paths} preds={live_pred_count}", flush=True)

    for batch in loader:
        bev = batch["bev"].to(device, non_blocking=use_cuda)
        origins = batch["origin"]
        with torch.inference_mode():
            heats = model(bev).squeeze(1).cpu().numpy()
        for j in range(heats.shape[0]):
            if idx >= len(paths):
                break
            idx += 1
            x0, y0 = float(origins[j, 0]), float(origins[j, 1])
            cid = chip_id(x0, y0)
            peaks = extract_peaks_from_heatmap(
                heats[j],
                origin_x=x0,
                origin_y=y0,
                cell_size=CELL_SIZE_M,
                score_threshold=score_threshold,
            )
            if live_dir is not None:
                pending_chips.append((cid, peaks))
            else:
                all_chips.append((cid, peaks))
        done_local = idx
        done = resume_done + done_local
        if progress_every > 0 and (
            done_local % progress_every == 0 or done_local == len(paths)
        ):
            print(f"infer chips {done}/{total_paths}", flush=True)
            if live_dir is not None:
                write_status(
                    live_dir / "status.json",
                    state="running",
                    done=done,
                    total=total_paths,
                    pred_count=live_pred_count,
                    model_id=model_id,
                )
        if live_dir is not None and done_local > 0:
            crossed = done // live_flush_every > last_live_flush // live_flush_every
            if crossed or done_local == len(paths):
                _flush_live(done, final=done_local == len(paths))

    if live_dir is not None and pending_chips:
        _flush_live(resume_done + idx, final=True)

    if live_dir is not None:
        merged = committed_merged
    else:
        merged = merge_chip_detections(all_chips, radius_m=merge_radius_m)
    print(f"merged predictions: {len(merged)}", flush=True)

    if verifier_checkpoint is not None:
        from treelidarnet.verifier import score_detections

        print(f"verifier: {verifier_checkpoint}", flush=True)
        merged = score_detections(
            merged,
            checkpoint=verifier_checkpoint,
            radius_m=verifier_radius_m,
        )
        if verifier_threshold is not None:
            before = len(merged)
            merged = [
                d
                for d in merged
                if d.verifier_score is not None and d.verifier_score >= verifier_threshold
            ]
            print(
                f"verifier threshold {verifier_threshold}: {len(merged)}/{before} kept",
                flush=True,
            )

    gts = load_ground_truths(trees_parquet, eligible_only=True)
    metrics = match_detections(merged, gts)

    df = pd.read_parquet(trees_parquet)
    df = df[df["eligible"]]
    gt_rows = df.to_dict(orient="records")

    export_path = export_path or (
        exports_dir() / "inference" / f"gent_city_{model_id}.json.gz"
    )
    export_city_gz(
        export_path,
        model_id=model_id,
        gt_rows=gt_rows,
        predictions=merged,
        metrics=metrics,
        chip_count=total_paths,
        peak_extract_min_score=score_threshold,
    )
    print(
        f"metrics audit F1={metrics.f1_audit:.3f} strict F1={metrics.f1:.3f} "
        f"discoveries={metrics.discoveries}",
        flush=True,
    )
    print(f"export: {export_path}", flush=True)
    if live_dir is not None:
        write_live_predictions(live_dir / "predictions.json", merged)
        write_status(
            live_dir / "status.json",
            state="done",
            done=total_paths,
            total=total_paths,
            pred_count=len(merged),
            model_id=model_id,
            metrics={
                "f1_at_3m": metrics.f1,
                "f1_audit_at_3m": metrics.f1_audit,
                "discoveries": metrics.discoveries,
            },
            export_gz=str(export_path),
        )
    return export_path

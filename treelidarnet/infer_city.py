"""Run model on all Gent chips; export city-scale Deck.GL bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from treelidarnet.data.chips import CELL_SIZE_M, chip_id
from treelidarnet.datasets.chip_dataset import ChipDataset, collate_chips
from treelidarnet.datasets.chip_index import chip_paths_outward_from_index, order_paths_outward
from treelidarnet.eval import load_ground_truths
from treelidarnet.export_mapjson import export_city_gz
from treelidarnet.infer import load_model
from treelidarnet.live_export import export_gt_gz, write_live_predictions, write_status
from treelidarnet.metrics import match_detections
from treelidarnet.paths import chips_dir, exports_dir
from treelidarnet.postprocess import (
    DEFAULT_PEAK_SCORE_THRESHOLD,
    Detection,
    extract_peaks_from_heatmap,
    merge_chip_detections,
)


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
    live_flush_every: int = 150,
    progress_every: int = 50,
    score_threshold: float = DEFAULT_PEAK_SCORE_THRESHOLD,
) -> Path:
    if chips_index is not None and chips_index.is_file():
        paths, center = chip_paths_outward_from_index(chips_index)
    else:
        paths, center = order_paths_outward(sorted(chips_dir().glob(chip_glob)))
    if not paths:
        raise FileNotFoundError(f"No chips in {chips_dir()}")
    if center is not None:
        print(
            f"infer order: outward from ({center[0]:.0f}, {center[1]:.0f})",
            flush=True,
        )
    if limit is not None:
        paths = paths[:limit]

    live_dir = exports_dir() / "inference" / "live" if live else None
    if live_dir is not None:
        df = pd.read_parquet(trees_parquet)
        df = df[df["eligible"]]
        export_gt_gz(live_dir / "gt.json.gz", df.to_dict(orient="records"))
        write_status(
            live_dir / "status.json",
            state="running",
            done=0,
            total=len(paths),
            pred_count=0,
            model_id=model_id,
        )
        print(f"live map: poll viewer/ghent_live.html (data under {live_dir})", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"infer {len(paths)} chips batch={batch_size} device={device} "
        f"min_confidence={score_threshold}",
        flush=True,
    )
    model = load_model(checkpoint, device)
    ds = ChipDataset(paths)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_chips,
        num_workers=0,
    )

    chip_detections: list[tuple[str, list[Detection]]] = []
    idx = 0
    live_pred_count = 0
    last_live_flush = 0
    for batch in loader:
        bev = batch["bev"].to(device)
        with torch.no_grad():
            heats = model(bev).squeeze(1).cpu().numpy()
        for j in range(heats.shape[0]):
            if idx >= len(paths):
                break
            npz_path = paths[idx]
            idx += 1
            origin = np.load(npz_path)["origin"]
            x0, y0 = float(origin[0]), float(origin[1])
            cid = chip_id(x0, y0)
            peaks = extract_peaks_from_heatmap(
                heats[j],
                origin_x=x0,
                origin_y=y0,
                cell_size=CELL_SIZE_M,
                score_threshold=score_threshold,
            )
            chip_detections.append((cid, peaks))
        done = idx
        if progress_every > 0 and (done % progress_every == 0 or done == len(paths)):
            print(f"infer chips {done}/{len(paths)}", flush=True)
            if live_dir is not None:
                write_status(
                    live_dir / "status.json",
                    state="running",
                    done=done,
                    total=len(paths),
                    pred_count=live_pred_count,
                    model_id=model_id,
                )
        # Flush when crossing each live_flush_every boundary (batch steps may skip exact multiples).
        if live_dir is not None and done > 0:
            crossed = done // live_flush_every > last_live_flush // live_flush_every
            if crossed or done == len(paths):
                partial = merge_chip_detections(chip_detections, radius_m=2.5)
                live_pred_count = len(partial)
                last_live_flush = done
                write_live_predictions(live_dir / "predictions.json", partial)
                write_status(
                    live_dir / "status.json",
                    state="running" if done < len(paths) else "merging",
                    done=done,
                    total=len(paths),
                    pred_count=live_pred_count,
                    model_id=model_id,
                )
                print(f"  live flush {done}/{len(paths)} preds={live_pred_count}", flush=True)

    merged = merge_chip_detections(chip_detections, radius_m=2.5)
    print(f"merged predictions: {len(merged)}", flush=True)

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
        chip_count=len(paths),
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
            done=len(paths),
            total=len(paths),
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

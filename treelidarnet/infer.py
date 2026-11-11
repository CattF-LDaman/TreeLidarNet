"""Run model on one chip; export map JSON."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from treelidarnet.data.chips import CELL_SIZE_M, grid_cells
from treelidarnet.datasets.chip_dataset import ChipDataset
from treelidarnet.export_mapjson import export_inference_gz
from treelidarnet.metrics import GroundTruth, match_detections
from treelidarnet.models.gent_detector import GentTreeDetector
from treelidarnet.paths import chips_dir, exports_dir
from treelidarnet.postprocess import DEFAULT_PEAK_SCORE_THRESHOLD, extract_peaks_from_heatmap


def load_model(checkpoint: Path, device: torch.device) -> GentTreeDetector:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    in_ch = int(ckpt.get("in_channels", 4))
    model = GentTreeDetector(in_channels=in_ch)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def infer_chip(
    *,
    checkpoint: Path,
    chip_id: str,
    trees_parquet: Path,
    export_path: Path | None = None,
    model_id: str = "gent_bev_v1",
    score_threshold: float = DEFAULT_PEAK_SCORE_THRESHOLD,
) -> Path:
    npz = chips_dir() / f"{chip_id}.npz"
    if not npz.is_file():
        raise FileNotFoundError(npz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    ds = ChipDataset([npz], labels=False)
    batch = ds[0]
    bev = batch["bev"].unsqueeze(0).to(device)
    with torch.inference_mode():
        heat = model(bev).squeeze(0).squeeze(0).cpu().numpy()
    origin = batch["origin"]
    x0, y0 = float(origin[0]), float(origin[1])
    preds = extract_peaks_from_heatmap(
        heat,
        origin_x=x0,
        origin_y=y0,
        cell_size=CELL_SIZE_M,
        score_threshold=score_threshold,
    )
    df = pd.read_parquet(trees_parquet)
    chip_size = grid_cells() * CELL_SIZE_M
    m = (
        df["eligible"]
        & (df["x"] >= x0)
        & (df["x"] < x0 + chip_size)
        & (df["y"] >= y0)
        & (df["y"] < y0 + chip_size)
    )
    gt_rows = df.loc[m].to_dict(orient="records")
    gts = [GroundTruth(gis_id=r["gis_id"], x=r["x"], y=r["y"]) for r in gt_rows]
    metrics = match_detections(preds, gts) if gts else None
    export_path = export_path or exports_dir() / "inference" / f"{chip_id}.json.gz"
    export_inference_gz(
        export_path,
        chip_id=chip_id,
        model_id=model_id,
        gt_rows=gt_rows,
        predictions=preds,
        metrics=metrics,
    )
    return export_path

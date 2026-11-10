"""Torch Dataset over chip npz files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from treelidarnet.data.chips import CELL_SIZE_M, grid_cells
from treelidarnet.models.bev_voxel import scatter_points_to_bev


class ChipDataset(Dataset):
    def __init__(self, chip_paths: list[Path], *, grid_size: int | None = None) -> None:
        self.paths = chip_paths
        self.grid_size = grid_size or grid_cells()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.paths[idx])
        xyz = torch.from_numpy(data["xyz"].astype(np.float32))
        feat = torch.from_numpy(data["feat"].astype(np.float32).copy())
        # DHMV intensity is uint16-scale; keep BEV in ~[0, 1] for stable fp16 convs.
        if feat.shape[0]:
            feat[:, 0] = feat[:, 0] / 65535.0
            feat[:, 1] = feat[:, 1] / 7.0
            feat[:, 2] = feat[:, 2] / 31.0
        heatmap = torch.from_numpy(data["heatmap"].astype(np.float32))
        ignore = torch.from_numpy(data["ignore"].astype(np.float32))
        if "unknown" in data:
            unknown = torch.from_numpy(data["unknown"].astype(np.float32))
        else:
            # Legacy chips: treat low-heatmap cells as unlabeled (canopy v2 fallback).
            heat_np = data["heatmap"].astype(np.float32)
            unknown = torch.from_numpy((heat_np < 0.05).astype(np.float32))
        if "hard_negative" in data:
            hard_negative = torch.from_numpy(data["hard_negative"].astype(np.float32))
        else:
            hard_negative = torch.zeros_like(heatmap)
        if xyz.shape[0] == 0:
            bev = torch.zeros(feat.shape[1] + 1, self.grid_size, self.grid_size)
        else:
            bev = scatter_points_to_bev(
                xyz,
                feat,
                grid_size=self.grid_size,
                cell_size=CELL_SIZE_M,
            )
        return {
            "bev": bev,
            "heatmap": heatmap.unsqueeze(0),
            "ignore": ignore.unsqueeze(0),
            "unknown": unknown.unsqueeze(0),
            "hard_negative": hard_negative.unsqueeze(0),
        }


def collate_chips(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "bev": torch.stack([b["bev"] for b in batch]),
        "heatmap": torch.stack([b["heatmap"] for b in batch]),
        "ignore": torch.stack([b["ignore"] for b in batch]),
        "unknown": torch.stack([b["unknown"] for b in batch]),
        "hard_negative": torch.stack([b["hard_negative"] for b in batch]),
    }

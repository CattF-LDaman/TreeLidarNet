"""Dataset over crops_index.parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from treelidarnet.data.crops import IN_CHANNELS, extract_crop_from_chip_path
from treelidarnet.paths import chips_dir, manifests_dir


class CropDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str | None = None,
        radius_m: float = 8.0,
        num_points: int = 512,
        min_points: int = 32,
        seed: int = 42,
        in_channels: int = IN_CHANNELS,
        explorer_fp_loss_weight: float = 1.0,
        explorer_fp_sample_weight: float = 1.0,
    ) -> None:
        df = pd.read_parquet(manifest_path)
        if split is not None:
            df = df[df["split"] == split]
        self.rows = df.to_dict("records")
        self._explorer_fp_loss_weight = float(explorer_fp_loss_weight)
        self._explorer_fp_sample_weight = float(explorer_fp_sample_weight)
        self.radius_m = radius_m
        self.num_points = num_points
        self.min_points = min_points
        self.in_channels = in_channels
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        chip_path = chips_dir() / f"{row['chip_id']}.npz"
        seed = hash(row["crop_id"]) % (2**31)
        rng = np.random.default_rng(seed)
        crop = extract_crop_from_chip_path(
            chip_path,
            center_x=float(row["center_x"]),
            center_y=float(row["center_y"]),
            radius_m=self.radius_m,
            num_points=self.num_points,
            min_points=self.min_points,
            rng=rng,
            in_channels=self.in_channels,
        )
        if crop is None:
            return None
        src = str(row.get("source") or "")
        is_fp = src == "explorer_fp"
        loss_w = self._explorer_fp_loss_weight if is_fp else 1.0
        return {
            "points": torch.from_numpy(crop),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "crop_id": row["crop_id"],
            "loss_weight": torch.tensor(loss_w, dtype=torch.float32),
        }

    def sample_weights(self) -> torch.Tensor:
        """Per-index weights for WeightedRandomSampler (train only)."""
        out: list[float] = []
        for row in self.rows:
            src = str(row.get("source") or "")
            out.append(
                self._explorer_fp_sample_weight if src == "explorer_fp" else 1.0
            )
        return torch.tensor(out, dtype=torch.double)


def collate_crops(batch: list[dict | None]) -> dict[str, torch.Tensor | list[str]] | None:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "points": torch.stack([b["points"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "loss_weight": torch.stack([b["loss_weight"] for b in batch]),
        "crop_id": [str(b["crop_id"]) for b in batch],
    }


def default_crop_manifest() -> Path:
    return manifests_dir() / "crops_index.parquet"
"""PointNet++ binary crop verifier."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from treelidarnet.data.crops import IN_CHANNELS
from treelidarnet.models.pointnet2_ops import PointNetSetAbstraction

MODEL_TYPE = "gent_crop_verifier"


class GentCropVerifier(nn.Module):
    """Two-scale PointNet++ tree / not-tree on fixed-size crops."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        sa1_npoint: int = 128,
        sa1_radius: float = 0.5,
        sa1_nsample: int = 32,
        sa2_npoint: int = 32,
        sa2_radius: float = 2.0,
        sa2_nsample: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        feat_ch = max(0, in_channels - 3)
        self.sa1 = PointNetSetAbstraction(
            sa1_npoint,
            sa1_radius,
            sa1_nsample,
            in_channel=3 + feat_ch,
            mlp=[64, 64, 128],
        )
        self.sa2 = PointNetSetAbstraction(
            sa2_npoint,
            sa2_radius,
            sa2_nsample,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: B, N, in_channels -> B, 1 logits (pre-sigmoid)."""
        xyz = x[:, :, :3]
        feat = x[:, :, 3:] if x.shape[-1] > 3 else None
        xyz1, feat1 = self.sa1(xyz, feat)
        _, feat2 = self.sa2(xyz1, feat1)
        pooled = torch.max(feat2, dim=1)[0]
        return self.head(pooled)


def load_crop_verifier(checkpoint: Path, device: torch.device) -> GentCropVerifier:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    arch = ckpt.get("arch") or {}
    model = GentCropVerifier(
        in_channels=int(arch.get("in_channels", IN_CHANNELS)),
        sa1_npoint=int(arch.get("sa1_npoint", 128)),
        sa1_radius=float(arch.get("sa1_radius", 0.5)),
        sa1_nsample=int(arch.get("sa1_nsample", 32)),
        sa2_npoint=int(arch.get("sa2_npoint", 32)),
        sa2_radius=float(arch.get("sa2_radius", 2.0)),
        sa2_nsample=int(arch.get("sa2_nsample", 64)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model
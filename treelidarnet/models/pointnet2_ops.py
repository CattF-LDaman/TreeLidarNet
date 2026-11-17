"""PointNet++ set-abstraction ops (pure PyTorch)."""

from __future__ import annotations

import torch
import torch.nn as nn


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """B, N, C x B, M, C -> B, N, M."""
    return torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)) ** 2, dim=-1)


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """B, N, 3 -> B, npoint index."""
    device = xyz.device
    b, n, _ = xyz.shape
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.ones(b, n, device=device) * 1e10
    farthest = torch.randint(0, n, (b,), dtype=torch.long, device=device)
    batch_indices = torch.arange(b, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """points B,N,C; idx B,S or B,S,K -> gather."""
    raw_size = idx.size()
    b, n, c = points.shape
    idx_flat = idx.reshape(b, -1)
    batch_idx = torch.arange(b, device=points.device).view(b, 1).expand_as(idx_flat)
    out = points[batch_idx, idx_flat, :]
    return out.view(*raw_size, c)


def ball_query(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """B,N,3 and B,S,3 -> B,S,nsample neighbor indices."""
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k=nsample, dim=-1, largest=False, sorted=False)
    r2 = radius * radius
    dists = torch.gather(sqrdists, 2, group_idx)
    fallback = group_idx[:, :, 0:1].expand(-1, -1, nsample)
    group_idx = torch.where(dists <= r2, group_idx, fallback)
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return new_xyz B,S,3 and new_points B,S,nsample,C."""
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = ball_query(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list[int],
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        last = in_channel
        layers: list[nn.Module] = []
        for out_ch in mlp:
            layers.extend(
                [
                    nn.Conv2d(last, out_ch, 1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            last = out_ch
        self.mlp_convs = nn.Sequential(*layers)

    def forward(
        self, xyz: torch.Tensor, points: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """xyz B,N,3; points B,N,C -> new_xyz B,S,3 new_feat B,S,C'."""
        new_xyz, grouped = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # B, S, nsample, Cin -> B, Cin, nsample, S
        grouped = grouped.permute(0, 3, 2, 1)
        new_feat = self.mlp_convs(grouped)
        new_feat = torch.max(new_feat, 2)[0]
        new_feat = new_feat.permute(0, 2, 1)
        return new_xyz, new_feat
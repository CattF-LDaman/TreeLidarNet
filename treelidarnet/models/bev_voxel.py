"""Dense voxel scatter → BEV tensor."""

from __future__ import annotations

import torch


def scatter_points_to_bev(
    xyz: torch.Tensor,
    feat: torch.Tensor,
    *,
    grid_size: int,
    cell_size: float,
) -> torch.Tensor:
    """Single item: xyz (N,3) local meters, feat (N,F). Returns (F+1, H, W)."""
    n_feat = feat.shape[1]
    out = torch.zeros(n_feat, grid_size, grid_size, dtype=xyz.dtype)
    cnt = torch.zeros(grid_size, grid_size, dtype=xyz.dtype)
    if xyz.shape[0] == 0:
        return torch.cat([out, cnt.unsqueeze(0)], dim=0)
    ix = (xyz[:, 0] / cell_size).long().clamp(0, grid_size - 1)
    iy = (xyz[:, 1] / cell_size).long().clamp(0, grid_size - 1)
    for i in range(xyz.shape[0]):
        out[:, iy[i], ix[i]] += feat[i]
        cnt[iy[i], ix[i]] += 1
    valid = cnt > 0
    if valid.any():
        out[:, valid] = out[:, valid] / cnt[valid]
    density = cnt / cnt.max().clamp(min=1.0)
    return torch.cat([out, density.unsqueeze(0)], dim=0)

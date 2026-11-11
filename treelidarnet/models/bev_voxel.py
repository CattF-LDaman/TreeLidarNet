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
    """Single item: xyz (N,3) local meters, feat (N,F). Returns (F+1, H, W).

    Sums points in stored order. The cell means match a per-point loop.
    """
    n_feat = feat.shape[1]
    out = torch.zeros(n_feat, grid_size, grid_size, dtype=xyz.dtype, device=xyz.device)
    cnt = torch.zeros(grid_size, grid_size, dtype=xyz.dtype, device=xyz.device)
    if xyz.shape[0] == 0:
        return torch.cat([out, cnt.unsqueeze(0)], dim=0)
    ix = (xyz[:, 0] / cell_size).long().clamp(0, grid_size - 1)
    iy = (xyz[:, 1] / cell_size).long().clamp(0, grid_size - 1)
    flat = iy * grid_size + ix
    ones = torch.ones(xyz.shape[0], dtype=xyz.dtype, device=xyz.device)
    cnt.view(-1).index_add_(0, flat, ones)
    out.view(n_feat, -1).index_add_(1, flat, feat.transpose(0, 1).contiguous())
    valid = cnt > 0
    if valid.any():
        out[:, valid] = out[:, valid] / cnt[valid]
    density = cnt / cnt.max().clamp(min=1.0)
    return torch.cat([out, density.unsqueeze(0)], dim=0)

import numpy as np
import torch

from treelidarnet.models.bev_voxel import scatter_points_to_bev


def _loop_scatter(
    xyz: torch.Tensor,
    feat: torch.Tensor,
    grid_size: int,
    cell_size: float,
) -> torch.Tensor:
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
    out[:, valid] = out[:, valid] / cnt[valid]
    density = cnt / cnt.max().clamp(min=1.0)
    return torch.cat([out, density.unsqueeze(0)], dim=0)


def test_scatter_matches_point_order_sum():
    torch.manual_seed(0)
    xyz = torch.rand(4000, 3) * 50
    xyz[:1000, :2] = xyz[1000:2000, :2]
    feat = torch.rand(4000, 3)
    got = scatter_points_to_bev(xyz, feat, grid_size=200, cell_size=0.25)
    ref = _loop_scatter(xyz, feat, 200, 0.25)
    assert torch.equal(got, ref)


def test_scatter_empty_is_zero():
    xyz = torch.zeros(0, 3)
    feat = torch.zeros(0, 3)
    got = scatter_points_to_bev(xyz, feat, grid_size=8, cell_size=0.25)
    assert got.shape == (4, 8, 8)
    assert torch.equal(got, torch.zeros_like(got))


def test_infer_item_keeps_the_training_grid(tmp_path):
    from treelidarnet.datasets.chip_dataset import ChipDataset

    n = 8
    path = tmp_path / "gent_50m_0_0.npz"
    rng = np.random.default_rng(0)
    xyz = rng.random((30, 3), dtype=np.float32) * 2
    feat = rng.integers(0, 100, size=(30, 3)).astype(np.float32)
    heat = np.zeros((n, n), dtype=np.float32)
    origin = np.array([1000.0, 2000.0], dtype=np.float64)
    np.savez_compressed(
        path,
        xyz=xyz,
        feat=feat,
        heatmap=heat,
        ignore=np.zeros((n, n), dtype=np.uint8),
        unknown=np.ones((n, n), dtype=np.float32),
        hard_negative=np.zeros((n, n), dtype=np.float32),
        origin=origin,
    )
    full = ChipDataset([path], grid_size=n)[0]
    infer = ChipDataset([path], grid_size=n, labels=False)[0]
    assert torch.equal(full["bev"], infer["bev"])
    assert "heatmap" not in infer
    assert torch.equal(infer["origin"], torch.from_numpy(origin))

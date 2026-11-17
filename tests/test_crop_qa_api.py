import numpy as np
import pandas as pd
from starlette.testclient import TestClient

from treelidarnet.crop_qa_server import CropQAServerConfig, create_crop_qa_app
from treelidarnet.data.chips import chip_id
from treelidarnet.paths import chips_dir, processed_dir


def test_crop_api_roundtrip(monkeypatch, tmp_path):
    chip_store = tmp_path / "chips"
    chip_store.mkdir()
    monkeypatch.setattr("treelidarnet.data.crops.chips_dir", lambda: chip_store)

    x0, y0 = 103_500.0, 195_800.0
    gx, gy = 103_517.3, 195_842.1
    local_x, local_y = gx - x0, gy - y0
    n = 400
    rng = np.random.default_rng(0)
    xyz = np.zeros((n, 3), dtype=np.float32)
    xyz[:, 0] = local_x + rng.uniform(-4, 4, size=n)
    xyz[:, 1] = local_y + rng.uniform(-4, 4, size=n)
    xyz[:, 2] = rng.uniform(2, 18, size=n)
    feat = np.stack(
        [
            rng.integers(0, 4000, size=n),
            np.ones(n),
            np.full(n, 4.0),
        ],
        axis=1,
    ).astype(np.float32)
    cid = chip_id(x0, y0)
    np.savez_compressed(
        chip_store / f"{cid}.npz",
        xyz=xyz,
        feat=feat,
        heatmap=np.zeros((200, 200), dtype=np.float32),
        ignore=np.zeros((200, 200), dtype=np.float32),
        unknown=np.zeros((200, 200), dtype=np.float32),
        origin=np.array([x0, y0], dtype=np.float64),
    )

    trees = pd.DataFrame(
        [
            {
                "gis_id": "test/tree1",
                "gis_slug": "test-tree1",
                "sortiment": "Test",
                "lon": 3.725,
                "lat": 51.054,
                "x": gx,
                "y": gy,
                "eligible": True,
                "stamomtrek": 80.0,
            }
        ]
    )
    trees_path = tmp_path / "trees.parquet"
    trees.to_parquet(trees_path, index=False)

    cfg = CropQAServerConfig(trees_parquet=trees_path, predictions_export=tmp_path / "none.gz")
    client = TestClient(create_crop_qa_app(cfg))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    crop = client.get("/api/crop", params={"x": gx, "y": gy})
    assert crop.status_code == 200
    body = crop.json()
    assert body["chip_id"] == cid
    assert len(body["points"]) == 512

    miss = client.get("/api/crop", params={"x": 1.0, "y": 1.0})
    assert miss.status_code == 404

    by_id = client.get("/api/crop", params={"gis_id": "test/tree1"})
    assert by_id.status_code == 200
    assert by_id.json()["gis_id"] == "test/tree1"

    reg = client.get("/api/register/all")
    assert reg.status_code == 200
    body = reg.json()
    assert body["count"] == 1
    assert body["trees"][0]["gis_id"] == "test/tree1"


def test_chips_dir_fixture_exists_or_skip():
    """Sanity: real chips_dir path is under processed (not used in API test)."""
    assert chips_dir() == processed_dir() / "chips"
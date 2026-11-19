import numpy as np
import pandas as pd

from treelidarnet.datasets.crop_dataset import CropDataset, collate_crops


def test_explorer_fp_loss_weight_in_batch(tmp_path, monkeypatch):
    chip_store = tmp_path / "chips"
    chip_store.mkdir()
    monkeypatch.setattr("treelidarnet.datasets.crop_dataset.chips_dir", lambda: chip_store)
    monkeypatch.setattr("treelidarnet.data.crops.chips_dir", lambda: chip_store)

    x0, y0 = 100_000.0, 190_000.0
    np.savez_compressed(
        chip_store / "gent_50m_100000_190000.npz",
        xyz=np.random.default_rng(0).uniform(15, 35, (120, 3)).astype(np.float32),
        feat=np.zeros((120, 3), dtype=np.float32),
        origin=np.array([x0, y0], dtype=np.float64),
    )

    manifest = tmp_path / "crops.parquet"
    pd.DataFrame(
        [
            {
                "crop_id": "fp_test",
                "label": 0,
                "source": "explorer_fp",
                "chip_id": "gent_50m_100000_190000",
                "center_x": 25.0,
                "center_y": 25.0,
                "split": "train",
            },
            {
                "crop_id": "grb_test",
                "label": 0,
                "source": "grb",
                "chip_id": "gent_50m_100000_190000",
                "center_x": 30.0,
                "center_y": 30.0,
                "split": "train",
            },
        ]
    ).to_parquet(manifest, index=False)

    ds = CropDataset(
        manifest,
        split="train",
        explorer_fp_loss_weight=15.0,
        explorer_fp_sample_weight=10.0,
    )
    a = ds[0]
    b = ds[1]
    assert a is not None and b is not None
    assert float(a["loss_weight"]) == 15.0
    assert float(b["loss_weight"]) == 1.0
    batch = collate_crops([a, b])
    assert batch is not None
    assert batch["loss_weight"].tolist() == [15.0, 1.0]
    sw = ds.sample_weights()
    assert sw[0].item() == 10.0
    assert sw[1].item() == 1.0